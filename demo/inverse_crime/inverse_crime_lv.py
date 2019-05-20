import os
from collections import OrderedDict
import dolfin
import pulse
import pulse_adjoint
from dolfin_adjoint import adj_reset
import matplotlib.pyplot as plt
import yaml
import numpy as np


def load_data():

    geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths["ellipsoid"])
    dolfin.File("cfun.pvd") << geometry.cfun
    matparams = pulse.HolzapfelOgden.default_parameters()
    pressures = [0.0, 1.0, 10.0, 12.0, 0.5, 1.0]
    gammas = [0.0, 0.0, 0.19, 0.22, 0.05, 0.0]
    return geometry, matparams, pressures, gammas


def group_in_file(fname, group):
    exist = False
    try:
        import h5py

        with h5py.File(fname, "r") as f:
            if group in f:
                exist = True
    finally:
        return exist


def build_problem(geometry, matparams):
    activation = dolfin.Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
    material = pulse.HolzapfelOgden(
        activation=activation,
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    # LV Pressure
    lvp = dolfin.Constant(0.0)
    lv_marker = geometry.markers["ENDO"][0]
    lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
    neumann_bc = [lv_pressure]

    # Add spring term at the base with stiffness 1.0 kPa/cm^2
    base_spring = 1.0
    robin_bc = [
        pulse.RobinBC(
            value=dolfin.Constant(base_spring), marker=geometry.markers["BASE"][0]
        )
    ]

    # 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
    def fix_basal_plane(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        bc = dolfin.DirichletBC(
            V.sub(0), dolfin.Constant(0.0), geometry.ffun, geometry.markers["BASE"][0]
        )
        return bc

    dirichlet_bc = [fix_basal_plane]

    # Collect boundary conditions
    bcs = pulse.BoundaryConditions(
        dirichlet=dirichlet_bc, neumann=neumann_bc, robin=robin_bc
    )

    # Create the problem
    problem = pulse.MechanicsProblem(geometry, material, bcs)
    return problem, activation, lvp


def forward(geometry, matparams, lvps, gammas, fname="forward_results.h5"):

    problem, activation, lvp = build_problem(geometry, matparams)

    us = []
    lv_volumes = []
    F_ref = None

    def compute_volumes(u):
        lvv = geometry.cavity_volume(u=u, chamber="lv")
        lv_volumes.append(lvv)
        print("LVP: {}, LVV: {}".format(plv, lvv))

    ED = next(i - 1 for i, g in enumerate(gammas) if np.all(g))
    us = []
    lv_volumes = []
    rv_volumes = []
    F_ref = None
    # Solve the problem
    for i, (plv, g) in enumerate(zip(lvps, gammas)):

        if group_in_file(fname, str(i)):
            u, p = problem.state.split(deepcopy=True)

            with dolfin.HDF5File(dolfin.mpi_comm_world(), fname, "r") as f:
                f.read(problem.state, "{}/state".format(i))

            u, p = dolfin.split(problem.state)

            lvp.assign(plv)
            activation.assign(dolfin.Constant(np.array(g)))
            try:
                problem.solve()
            except pulse.mechanicsproblem.SolverDidNotConverge:
                pass
            else:
                us.append(u)
                compute_volumes(u)
                if i == ED:
                    F_ref = pulse.DeformationGradient(u)
                continue

        pulse.iterate.iterate(problem, lvp, plv)
        pulse.iterate.iterate(problem, activation, g, max_iters=100)

        u, p = dolfin.split(problem.state)
        us.append(u)
        compute_volumes(u)

        file_mode = "a" if os.path.exists(fname) else "w"
        with dolfin.HDF5File(dolfin.mpi_comm_world(), fname, file_mode) as f:
            f.write(problem.state, "{}/state".format(i))

        if i == ED:
            F_ref = pulse.DeformationGradient(u)

    crl_basis = dict(
        circumferential=geometry.c0, radial=geometry.r0, longitudinal=geometry.l0
    )
    straintarget = pulse_adjoint.optimization_targets.RegionalStrainTarget(
        geometry.mesh, crl_basis, geometry.dx, tensor="E", F_ref=F_ref
    )
    straintarget.set_target_functions()

    nregions = straintarget.nregions
    strains = OrderedDict({"region_{}".format(i + 1): [] for i in range(nregions)})
    for u in us:
        straintarget.assign_simulated(u)
        for region in range(nregions):
            strains["region_{}".format(region + 1)].append(
                straintarget.simulated_fun[region].vector().get_local()
            )

    return dict(LVV=lv_volumes, LVP=lvps, strains=strains)


def inverse(geometry, data):

    pulse_adjoint.setup_parameters.setup_general_parameters()
    params = pulse_adjoint.setup_parameters.setup_adjoint_contraction_parameters()
    params["sim_file"] = "results_file.h5"
    pulse.parameters["log_level"] = 40
    # Select a different initial material parameter
    params["Material_parameters"]["a"] = 1.0
    data["passive_filling_duration"] = 2
    params["strain_reference"] = "ED"
    params["strain_tensor"] = "E"
    params["base_spring"] = 1.0
    params["active_model"] = "active_strain"
    params["gamma_space"] = "R_0"
    params["Optimization_parameters"]["gamma_max"] = 0.4

    pulse_adjoint.assimilate(geometry, data, params)
    return params


def plot_data(data):

    fig_vol, ax_vol = plt.subplots()
    fig_circ, ax_circ = plt.subplots(3, 6, sharey=True)
    fig_rad, ax_rad = plt.subplots(3, 6, sharey=True)
    fig_long, ax_long = plt.subplots(3, 6, sharey=True)
    ax_vol.set_xlabel("Volume (mL)")
    ax_vol.set_ylabel("Pressure (kPa)")
    for name, ax in zip(
        ["circumferential", "radial", "longitudinal"], [ax_circ, ax_rad, ax_long]
    ):
        for i, axi in enumerate(ax.flatten()):
            if i == 17:
                axi.axis("off")
            else:
                axi.set_title("Region {}".format(i + 1))
        for j in range(6):
            ax[2, j].set_xlabel("Time point")
        ax[1, 0].set_ylabel("{} strain".format(name))

    ax_vol.plot(data["LVV"], data["LVP"])

    for i, (ax, name) in enumerate(
        zip([ax_circ, ax_rad, ax_long], ["circumferential", "radial", "longitudinal"])
    ):
        for j, strain in enumerate(data["strains"].values()):
            ax.flatten()[j].plot(np.transpose(strain)[i])

    for name, fig in zip(
        ["pv_loop", "circumferential_strain", "radial_strain", "longitudinal_strain"],
        [fig_vol, fig_circ, fig_rad, fig_long],
    ):
        fig.tight_layout()
        fig.savefig(name)


def postprocess(params, data):

    pulse_adjoint.postprocess.PostProcess(fname=params['sim_file'],
                                          parameters=params,
                                          data=data)


def main():

    geo, matparams, lvps, gammas = load_data()
    data = forward(geo, matparams, lvps, gammas)
    # plot_data(data)
    params = inverse(geo, data)
    postprocess(params, data)


if __name__ == "__main__":
    main()
