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
from dolfin_adjoint import replay_dolfin, adj_reset, adj_html
import dolfin
from pulse_adjoint.run_optimization import run_passive_optimization_step, run_active_optimization_step, run_passive_optimization
from pulse_adjoint.setup_optimization import initialize_patient_data, setup_simulation
from pulse_adjoint.adjoint_contraction_args import *
from pulse_adjoint.utils import Text, pformat, passive_inflation_exists
from test_utils import setup_params, my_taylor_test, store_results, plot_displacements
def test_passive(params):
    
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
    
    
    logger.info(Text.blue("\nTest Passive Optimization"))

    logger.info(pformat(params.to_dict()))


    params["phase"] = "passive_inflation"
    # params["alpha_matparams"] = 0.5
    measurements, solver_parameters, p_lv, paramvec = \
      setup_simulation(params, patient)

    
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

    store_results(params, rd, {})
    


def test_active(params):
    
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
    
    logger.info(Text.blue("\nTest Passive Optimization"))

    logger.info(pformat(params.to_dict()))

    if not passive_inflation_exists(params):
        params["phase"] = "passive_inflation"
        params["optimize_matparams"] = False
        run_passive_optimization(params, patient)
        
    adj_reset()

    params["phase"] = "active_contraction"
    measurements, solver_parameters, p_lv, gamma = \
      setup_simulation(params, patient)
    
    rd, gamma = run_active_optimization_step(params, 
                                             patient, 
                                             solver_parameters, 
                                             measurements, p_lv, 
                                             gamma)

    # Test that we can store the results
    store_results(params, rd, gamma)

    

    # Dump html visualization of the forward and adjoint system
    adj_html("active_forward.html", "forward")
    adj_html("active_adjoint.html", "adjoint")

    # Replay the forward run, i.e make sure that the recording is correct.
    logger.info("Replay dolfin")
    assert replay_dolfin(tol=1e-12)
    
    # Test that the gradient is correct
    logger.info("Taylor test")
    my_taylor_test(rd, gamma)

    
def test_lv():

    from itertools import product
    spaces = ["CG_1", "regional", "R_0"]

    opt_targets = ["volume", "regional_strain"]

    for space in spaces:
        params = setup_params(space, "lv", opt_targets)
        
        test_passive(params)
        test_active(params)
        
def test_biv():

    from itertools import product
    spaces = ["CG_1", "R_0"]

    opt_targets = ["volume", "rv_volume"]

    for space in spaces:
        params = setup_params(space, "biv", opt_targets)
        
        test_passive(params)
        test_active(params)
     


if __name__ == "__main__":
    # plot_displacements()
    # exit()
    # test_biv()
    test_lv()
    
   
