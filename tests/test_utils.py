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
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pulse_adjoint.adjoint_contraction_args import *
from pulse_adjoint.setup_optimization import setup_adjoint_contraction_parameters, setup_general_parameters, setup_passive_optimization_weigths, setup_active_optimization_weigths
from pulse_adjoint.numpy_mpi import *
from pulse_adjoint.utils import Text
def setup_params(space = "CG_1", mesh_type = "lv", opt_targets = ["volume"]):
    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    
    for key in params["Optimization_targets"].keys():
        if key in opt_targets:
            params["Optimization_targets"][key] = True
        else:
            params["Optimization_targets"][key] = False


    
    # Update weights
    pparams = setup_passive_optimization_weigths(params["Optimization_targets"])
    aparams = setup_active_optimization_weigths(params["Optimization_targets"])

    params.remove('Passive_optimization_weigths')
    params.add(pparams)
    params.remove('Active_optimization_weigths')
    params.add(aparams)
                
    params["gamma_space"] = space
    params["Patient_parameters"]["mesh_type"] = mesh_type
    params["Patient_parameters"]["patient"] = "test"
    params["Patient_parameters"]["patient_type"] = "test"
    params["sim_file"] = "data/test.h5"
    params["outdir"] = "data"
    set_log_active(True)

    logger.setLevel(DEBUG)

    return params


def my_taylor_test(Jhat, m0_fun):
    m0 = gather_broadcast(m0_fun.vector().array())
      
    Jm0 = Jhat(m0)
    DJm0 = Jhat.derivative(forget=False)

    d = np.array([1.0]*len(m0)) #perturbation direction
    grad_errors = []
    no_grad_errors = []
   
    epsilons = [0.05, 0.025, 0.0125]
        
    for eps in epsilons:
        m = np.array(m0 + eps*d)
        
        Jm = Jhat(m)
        no_grad_errors.append(abs(Jm - Jm0))
        grad_errors.append(abs(Jm - Jm0 - np.dot(DJm0, m - m0)))
       
    logger.info("Errors without gradient: {}".format(no_grad_errors))
    logger.info("Convergence orders without gradient (should be 1)")
    logger.info("{}".format(convergence_order(no_grad_errors)))
   
    logger.info("\nErrors with gradient: {}".format(grad_errors))
    logger.info("Convergence orders with gradient (should be 2)")
    con_ord = convergence_order(grad_errors)
    logger.info("{}".format(con_ord))
    
    assert (np.array(con_ord) > 1.85).all()


def store_results(params, rd, control):
    from pulse_adjoint.store_opt_results import write_opt_results_to_h5
    
    if params["phase"] == "passive_inflation":
        h5group =  PASSIVE_INFLATION_GROUP
        write_opt_results_to_h5(h5group, params, rd.ini_for_res, 
                                    rd.for_res, opt_matparams = control)
        
    else:
        h5group =  ACTIVE_CONTRACTION_GROUP.format(0)
        write_opt_results_to_h5(h5group, params, rd.ini_for_res, 
                                    rd.for_res, opt_gamma = control)


def plot_displacements():
    params = setup_params()
    params["base_bc"] =  "dirichlet_bcs_from_seg_base"#"dirichlet_bcs_fix_base_x"
    from pulse_adjoint.setup_optimization import initialize_patient_data
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])

    alpha_regpars = [(params["alpha"], params["reg_par"])]
    # from IPython import embed; embed()
    # exit()
    from pulse_adjoint.postprocessing.postprocess_utils import get_all_data
    data, kwargs = get_all_data(params, patient, alpha_regpars)

    
    u = Function(kwargs['displacement_space'], name = "displacement")
    us = data["passive"]["displacements"]

    f = XDMFFile(mpi_comm_world(), "data/displacement.xdmf")
    for it,u_ in enumerate(us):
        u.vector()[:] = u_
        f << u, float(it)
        
    
