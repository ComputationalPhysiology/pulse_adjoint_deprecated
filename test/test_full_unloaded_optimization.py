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
from utils import setup, set_patient_attributes
from pulse_adjoint.setup_optimization import setup_general_parameters
import os

phases = ["passive", "active"]
material_models = ["holzapfel_ogden", "neo_hookean"]
active_models = ["active_strain", "active_stress"]

# (endo, epi)


def optimize():


    phase = "passive"
    material_model = "holzapfel_ogden"
    
    active_model = "active_strain"
    fiber_angle = (60,-60)

    # (strain, volume, regularization)
    weight = (0.0,1.0,0.0)
    ndiv = 1
    control_regions = [2]
    strain_regions = [2]
    eps_strain = 0.0
    eps_vol = 0.0
    space = "R_0"
    unload = True
    pressures = [0, 0.6, 0.8]
    h5name = "test_unload_geometry_idx_0.h5"
    approx = "interpolate"
    geometry_index = "0"
    
    params, strains, vols, ap_params, p_lv, w, u, p, f_ex, basis, u_img \
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
               space = space, unload = unload,
               h5name=h5name, approx = approx,
               geometry_index = geometry_index,
               isotropic=True, restart = True)
    

    patient = set_patient_attributes(params, pressures, vols, strains, True)

    import numpy as np
    volume = np.transpose(vols)[1]
  
    ap_params["Unloading_parameters"]["estimate_initial_guess"] = True#False
    from pulse_adjoint.run_optimization import run_unloaded_optimization
    run_unloaded_optimization(ap_params, patient)
    # run_unloaded_optimization(params, strains, vols, ap_params,
                              # pressures,p_lv, return_rd = False)


    

   

if __name__ == "__main__":
    
    setup_general_parameters()
    optimize()
    
