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
from dolfin import parameters
from dolfin_adjoint import replay_dolfin, adj_reset, adj_html
import dolfin
from pulse_adjoint.run_optimization import run_passive_optimization_step, run_active_optimization_step, run_passive_optimization
from pulse_adjoint.setup_optimization import initialize_patient_data, setup_simulation
from pulse_adjoint.adjoint_contraction_args import *
from pulse_adjoint.utils import Text, pformat, passive_inflation_exists
from utils import setup_params, my_taylor_test, store_results, plot_displacements
from pulse_adjoint.numpy_mpi import *

import pytest
import itertools

parametrize = pytest.mark.parametrize

mesh_types = ["biv"]
spaces = ["regional", "CG_1"]
phases = ["passive", "active"]
active_models = ["active_strain", "active_stress"]




def passive(params):

    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
    
    
    logger.info(Text.blue("\nTest Passive Optimization"))

    logger.info(pformat(params.to_dict()))


    params["phase"] = "passive_inflation"
    measurements, solver_parameters, p_lv, paramvec \
        = setup_simulation(params, patient)

    
    rd, paramvec = run_passive_optimization_step(params, 
                                                 patient, 
                                                 solver_parameters, 
                                                 measurements, 
                                                 p_lv, paramvec)    
    
    
    # Dump html visualization of the forward and adjoint system
    adj_html("passive_forward.html", "forward")
    adj_html("passive_adjoint.html", "adjoint")
    
    # Replay the forward run, i.e make sure that the recording is correct.
    logger.info("Replay dolfin")
    assert replay_dolfin(tol=1e-12)
    
    # Test that the gradient is correct
    logger.info("Taylor test")
    my_taylor_test(rd, paramvec)

    paramvec_arr = gather_broadcast(paramvec.vector().array())
    rd(paramvec_arr)
    rd.for_res["initial_control"] = rd.initial_paramvec,
    rd.for_res["optimal_control"] = rd.paramvec
    store_results(params, rd, {})
    

def active(params):

   
    
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
    
    logger.info(Text.blue("\nTest Passive Optimization"))
    
    logger.info(pformat(params.to_dict()))
    
    if 1:#not passive_inflation_exists(params):
        params["phase"] = "passive_inflation"
        params["optimize_matparams"] = False
        run_passive_optimization(params, patient)
        
        adj_reset()

    print params["sim_file"]
    params["phase"] = "active_contraction"
    params["active_contraction_iteration_number"] = 0
    measurements, solver_parameters, p_lv, gamma \
        = setup_simulation(params, patient)

    
    dolfin.parameters["adjoint"]["test_derivative"] = True
     
    rd, gamma = run_active_optimization_step(params, 
                                             patient, 
                                             solver_parameters, 
                                             measurements, p_lv, 
                                             gamma)


    

    # Dump html visualization of the forward and adjoint system
    adj_html("active_forward.html", "forward")
    adj_html("active_adjoint.html", "adjoint")
        
    # Replay the forward run, i.e make sure that the recording is correct.
    logger.info("Replay dolfin")
    assert replay_dolfin(tol=1e-12)
    
    # Test that the gradient is correct
    logger.info("Taylor test")
    my_taylor_test(rd, gamma)
        
    paramvec_arr = gather_broadcast(gamma.vector().array())
    rd(paramvec_arr)
    rd.for_res["initial_control"] = rd.initial_paramvec,
    rd.for_res["optimal_control"] = rd.paramvec
    store_results(params, rd, {})
        
    

@parametrize(("mesh_type", "space", "phase", "active_model"),
             list(itertools.product(mesh_types, spaces, phases, active_models)))
def test_adjoint_calculations(mesh_type, space, phase, active_model):

    if mesh_type == "lv":
        opt_targets = ["volume", "regional_strain", "regularization"]
    else:
        opt_targets = ["volume", "rv_volume", "regularization"]

    params = setup_params(phase, space, mesh_type, opt_targets, active_model)

    if phase == "passive":
        passive(params)
        
    elif phase == "active":
        active(params)

    else:
        assert False
    
if __name__ == "__main__":
    test_adjoint_calculations("biv", "regional", "passive", "active_strain")
    
