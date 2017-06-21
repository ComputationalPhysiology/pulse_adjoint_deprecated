#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
from dolfin_adjoint import adj_reset
from .setup_optimization import (setup_adjoint_contraction_parameters,
                                 setup_general_parameters,
                                 initialize_patient_data,
                                 save_patient_data_to_simfile,
                                 update_unloaded_patient)

from .run_optimization import (run_passive_optimization,
                               run_active_optimization,
                               run_unloaded_optimization)

from .utils import  Text, pformat
from .io import passive_inflation_exists, contract_point_exists

from .adjoint_contraction_args import *
from .unloading import UnloadedMaterial

def save_logger(params):

    import os
    outdir = os.path.dirname(params["sim_file"])
    logfile = "output.log" if outdir == "" else outdir + "/output.log"    
    logging.basicConfig(filename=logfile,
                        filemode='a',
                        format='%(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    
    ffc_logger = logging.getLogger('FFC')
    ffc_logger.setLevel(logging.WARNING)
    ufl_logger = logging.getLogger('UFL')
    ufl_logger.setLevel(logging.WARNING)
 
    import datetime
    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    logger.info("Time: {}".format(time))
    

def main(params):

    save_logger(params)   
    
    setup_general_parameters()

    
    logger.info(Text.blue("Start Adjoint Contraction"))
    logger.info(pformat(params.to_dict()))
    logger.setLevel(params["log_level"])

    ############# GET PATIENT DATA ##################
    patient = initialize_patient_data(params["Patient_parameters"])
    
    
    # Save mesh and fibers to result file
    save_patient_data_to_simfile(patient, params["sim_file"])


    ############# RUN MATPARAMS OPTIMIZATION ##################
    
    # Make sure that we choose passive inflation phase
    params["phase"] =  PHASES[0]
    if not passive_inflation_exists(params):

        if params["unload"]:
            
            run_unloaded_optimization(params, patient)
           
        else:
            run_passive_optimization(params, patient)
            
        adj_reset()

    
    if params["unload"]:

        patient = update_unloaded_patient(params, patient)
        


    ################## RUN GAMMA OPTIMIZATION ###################

    # Make sure that we choose active contraction phase
    params["phase"] =  PHASES[1]
    run_active_optimization(params, patient)
   
        
if __name__ == '__main__':

    # parser = get_parser()
    # args = parser.parse_args()
    # main(args)
    
    params = setup_adjoint_contraction_parameters()
    main(params)
    
