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
from setup_optimization import setup_adjoint_contraction_parameters, setup_general_parameters, initialize_patient_data, save_patient_data_to_simfile
from run_optimization import run_passive_optimization, run_active_optimization
from adjoint_contraction_args import *
from utils import passive_inflation_exists, contract_point_exists,  Text, pformat
from dolfin_adjoint import adj_reset




def main(params):

    setup_general_parameters()
    

    logger.info(Text.blue("Start Adjoint Contraction"))
    logger.info(pformat(params.to_dict()))
    

    ############# GET PATIENT DATA ##################
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
      
    # Save mesh and fibers to result file
    save_patient_data_to_simfile(patient, params["sim_file"])


    ############# RUN MATPARAMS OPTIMIZATION ##################
    
    # Make sure that we choose passive inflation phase
    params["phase"] =  PHASES[0]
    if not passive_inflation_exists(params):
        run_passive_optimization(params, patient)
        adj_reset()


    
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
    
