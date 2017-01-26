#!/usr/bin/env python
# Copyright (C) 2016 Henrik Finsberg
#
# This file is part of PULSE-ADJOINT.
#
# PULSE-ADJOINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PULSE-ADJOINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PULSE-ADJOINT. If not, see <http://www.gnu.org/licenses/>.

from utils import *

phases = ["passive", "active"]
material_models = ["holzapfel_ogden", "neo_hookean"]
active_models = ["active_strain", "active_stress"]

# (endo, epi)
fiber_angles = [(0,0), (30,-30), (60,-60), (90, -90), (45,-45)]

# (strain, volume)
weights = [(1.0,0.0), (0.5, 0.5), (0.0,1.0)]


def make_solver_params(geo):

    
    ffun = geo.ffun
    mesh = geo.mesh

    N = FacetNormal(mesh)

    
    def base_bc(W):
        '''Fix the basal plane.
        '''
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        bc = [DirichletBC(V, Constant((0, 0, 0)), geo.markers["BASE"][0])]
        return bc


    
    V_real = FunctionSpace(mesh, "R", 0)
    p_lv = Expression("t", t = 0.0,
                      name = "LV_endo_pressure", element = V_real.ufl_element())

    neumann_bc = [[p_lv, geo.markers["ENDO"][0]]]
    pressure = {"p_lv":p_lv}

    
    nsolver = "snes_solver"
    prm = {"nonlinear_solver": "snes", "snes_solver":{}}
    
    
    prm[nsolver]['absolute_tolerance'] = 1E-8
    prm[nsolver]['relative_tolerance'] = 1E-8
    prm[nsolver]['maximum_iterations'] = 15
    prm[nsolver]['linear_solver'] = 'lu'
    prm[nsolver]['error_on_nonconvergence'] = True
    prm[nsolver]['report'] = False
        

    params= {"mesh": mesh,
             "facet_function": ffun,
             "passive_filling_duration": 1,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "solve":prm, 
             "bc":{"dirichlet": base_bc,
                   "neumann":neumann_bc}}

    return params, p_lv

def setup(phase, material_model, active_model,
          fiber_angle, weight, ndiv,
          control_regions, strain_regions):

    msg = ("phase = {}\n".format(phase) + \
           "material_model = {}\n".format(material_model) + \
           "active_model = {}\n".format(active_model) + \
           "fiber_angle (endo, epi) = {}\n".format(fiber_angle) + \
           "weights (strain, volume) = {}\n".format(weight) + \
           "control_regions = {}\n".format(control_regions) + \
           "strain_regions = {}\n".format(strain_regions))
    print(msg)

    # Material coefficients
    if material_model == "neo_hookean":
        mu = 0.385
        matparams = {"mu": mu}
    else:
        a = 1.0
        b = 1.0
        a_f = 1.0
        b_f = 1.0

        matparams = {"a":a, "b":b, "a_f":a_f, "b_f":b_f}


    # expr = Expression("sin(x[0])+2.0")
    if phase == "passive":
        expr = Expression("x[2]-x[1]+x[0]+2.0")
        initial_guess = 1.0
    else:
        expr =  Expression("0.1*(1.0+0.3*(x[2]+x[1]+x[0]))")
        initial_guess = 0.0
    



    geo = load_geometry_from_h5("data/mesh_simple_{}.h5".format(ndiv))
    mesh = geo.mesh
    foc = 1.54919333848

    

    
    # This is temporary
    control_markers = mark_strain_regions(mesh, foc, control_regions, mark_mesh =False)
    strain_markers =  mark_strain_regions(mesh, foc, strain_regions, mark_mesh =False)

    regions = sum(strain_regions)

    e_circ = geo.circumferential
    e_long = geo.longitudinal
    e_rad = geo.radial

    basis = {"e_circ":e_circ, "e_long": e_long, "e_rad": e_long}

    fiber_params = setup_fiber_parameters()
    fiber_params["fiber_angle_endo"] = fiber_angle[0]
    fiber_params["fiber_angle_epi"] = fiber_angle[1]
    fields = generate_fibers(mesh, fiber_params)
    f0 = fields[0]
    

    # Make the solver parameters
    params, p_lv = make_solver_params(geo)



    params["basis"] = basis
    params["f0"] = f0
    params["strain_markers"] = strain_markers
    params["control_markers"] = control_markers
    params["nregions"] = regions
    params["active_model"] = active_model
    params["material_model"] = material_model
    params["markers"] = geo.markers
    params["material_parameters"] = matparams
    params["dmu"] = Measure("dx", subdomain_data = strain_markers, domain = mesh)
    params["dmu_control"] = Measure("dx", subdomain_data = control_markers, domain = mesh)
    
    space = "regional"
    ap_params = get_application_parameters(space, phase, active_model)
    ap_params["Passive_optimization_weigths"]["regularization"] = 0.0
    ap_params["Passive_optimization_weigths"]["regional_strain"] = weight[0]
    ap_params["Passive_optimization_weigths"]["volume"] = weight[1]
    
    ap_params["Active_optimization_weigths"]["regularization"] = 0.0
    ap_params["Active_optimization_weigths"]["regional_strain"] = weight[0]
    ap_params["Active_optimization_weigths"]["volume"] = weight[1]
    
    ap_params["sim_file"] = "test_opt.h5"
   
   

    parameters["adjoint"]["stop_annotating"] = True
    strains, vols,  w, u, p, f_ex = generate_data(expr, params, ap_params, p_lv, phase)
    params, p_lv = make_solver_params(geo)
    params["basis"] = basis
    params["f0"] = f0
    params["strain_markers"] = strain_markers
    params["control_markers"] = control_markers
    params["nregions"] = regions
    params["markers"] = geo.markers
    params["active_model"] = active_model
    params["material_model"] = material_model
    params["material_parameters"] = matparams
    params["dmu"] = Measure("dx", subdomain_data = strain_markers, domain = mesh)
    params["dmu_control"] = Measure("dx", subdomain_data = control_markers, domain = mesh)
    parameters["adjoint"]["stop_annotating"] = False

    return params, strains, vols, ap_params, p_lv, w, u, p, f_ex, basis



def optimize():

    plot_sol =True
        
    phase = "active"
    material_model = "holzapfel_ogden"
    active_model = "active_strain"
    fiber_angle = (60,-60)
    weight = (0.5,0.5)
    ndiv = 1
    control_regions = [2]
    strain_regions = [1]


    params, strains, vols, ap_params, p_lv, w, u, p, f_ex, basis \
        =setup(phase = phase,
               material_model = material_model,
               active_model = active_model,
               fiber_angle = fiber_angle,
               weight = weight,
               ndiv = ndiv,
               control_regions = control_regions,
               strain_regions = strain_regions)
    
    print("Exact control = {}\n".format(f_ex.vector().array()))
    ap_params["passive_relax"] = 1.0
    ap_params["active_relax"] = 1.0
    rd,  forward_result, targets = run_optimization(params, strains, vols, ap_params, p_lv, 0.0, return_rd = False)
    
    
    sim_strains = np.array([gather_broadcast(v.array()) for v in targets["regional_strain"].results["simulated"][-1]])
    
    strains = np.array(strains)
    
    sim_vol = gather_broadcast(targets["volume"].results["simulated"][-1].array())[0]
    msg = ("\nSynthetic volume: {:.5f}".format(vols[-1]) + \
           "\nSimulated volume: {:.5f}".format(sim_vol) + \
           "\nDifference (abs): {:.2e}".format(abs(vols[-1]-sim_vol)))

    print(msg)
                                         

    if plot_sol:
        
        f, ax = plt.subplots(1, len(basis.keys()))
        
        for i, k in enumerate(basis.keys()):
            ks = k.split("_")[-1]
            ax[i].set_title(ks)
   
            x = range(len(strains.T[i]))
            ax[i].plot(x, strains.T[i], "bo", label = "synthetic")
            ax[i].plot(x, sim_strains.T[i],  "ro", label = "simulated")
            ax[i].legend()
        plt.show()

    if phase == "active":

        paramvec = get_optimal_gamma(params, ap_params)
        space_str = "gamma_space"
    else:
        
        paramvec = get_optimal_matparam(params, ap_params)
        space_str = "matparams_space"

    if ap_params[space_str] == "regional":
        V = FunctionSpace(params["mesh"], "DG", 0)
        f_DG = project(f_ex.get_function(), V)
        
        paramvec_DG = project(paramvec.get_function(), V)

        if plot_sol:
            
            plot(paramvec_DG, mode = "color")
            plot(f_DG, mode="color")
            plot(paramvec_DG-f_DG, mode = "color")
            interactive()
        
    else:
        
        if plot_sol:
        
            plot(paramvec, mode = "color")
            plot(f_ex, mode="color")
            interactive()
            plot(paramvec-f_ex, mode = "color")
            interactive()



if __name__ == "__main__":
    setup_general_parameters()
    optimize()
