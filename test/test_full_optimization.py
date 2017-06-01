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


def optimize(phase = "passive"):

    plot_sol =True

    material_model = "holzapfel_ogden"
    
    # active_model = "active_stress"
    active_model = "active_strain"
    fiber_angle = (60,-60)
    # weight = (0.0, 1.0)
    weight = (0.5, 0.5, 0.0)
    ndiv = 1
    control_regions = [2]
    strain_regions = [2]
    eps_strain = 0.0
    eps_vol = 0.0
    space = "regional"
    h5name = "test_full_optimization.h5"
    approx = "project"
    geometry_index = "0"
    
    if phase == "passive":
        pressures = [0, 0.4, 0.8, 1.2]
    else:
        pressures = [0, 1.0]

    params, strains, vols, ap_params, p_lv, w, u, p, f_ex, basis, _ \
        =setup(phase = phase,
               material_model = material_model,
               active_model = active_model,
               fiber_angle = fiber_angle,
               weight = weight,
               ndiv = ndiv,
               control_regions = control_regions,
               strain_regions = strain_regions,
               eps_vol = eps_vol,
               eps_strain = eps_strain,
               pressures = pressures,
               space = space, unload = False,
               h5name=h5name, approx = approx,
               geometry_index = geometry_index,
               isotropic=True, restart = False)
    
    
    print("Exact control = {}\n".format(f_ex.vector().array()))
    ap_params["passive_relax"] = 1.0
    ap_params["active_relax"] = 1.0
    initial_guess = 2.8 if phase == "passive" else 0.0
    rd,  forward_result, targets = run_optimization(params, strains, vols, ap_params, pressures,
                                                    p_lv["p_lv"], initial_guess, return_rd = False)
    

    print("Exact control = {}\n".format(f_ex.vector().array()))
    
    sim_strains = np.array([gather_broadcast(v.array()) for v in targets["regional_strain"].results["simulated"][-1]])
    
    strains = np.array(strains[-1])
    
    sim_vol = gather_broadcast(targets["volume"].results["simulated"][-1].array())[0]

    msg = ("\nSynthetic volume: {:.5f}".format(vols[-1][-1]) + \
           "\nSimulated volume: {:.5f}".format(sim_vol) + \
           "\nDifference (abs): {:.2e}".format(abs(vols[-1][-1]-sim_vol)))

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
        diff = project(paramvec_DG-f_DG, V)

        diff_max = norm(diff.vector(), 'linf')
        diff_l2 = norm(diff)

        msg = ("\nError in control: \n\tMax:\t{}\n\tL2:\t{}".format(diff_max, diff_l2))
        print(msg)

        if plot_sol:
            
            plot(paramvec_DG, mode = "color")
            plot(f_DG, mode="color")
            plot(diff, mode = "color")
            interactive()
        
    else:

        diff = project(paramvec-f_ex, paramvec.function_space())
        diff_max = norm(diff.vector(), 'linf')
        diff_l2 = norm(diff)

        msg = ("\nError in control: \n\tMax:\t{}\n\tL2:\t{}".format(diff_max, diff_l2))
        print(msg)

        if plot_sol:
        
            plot(paramvec, mode = "color")
            plot(f_ex, mode="color")
            interactive()
            plot(diff, mode = "color")
            interactive()



if __name__ == "__main__":
    setup_general_parameters()
    optimize("passive")
    optimize("active")
