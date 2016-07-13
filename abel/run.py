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
from pulse_adjoint.run_full_optimization import main as real_main
from pulse_adjoint.synthetic_data import run_active_synth_data as synth_main
from pulse_adjoint.adjoint_contraction_args import *
from pulse_adjoint.setup_optimization import setup_adjoint_contraction_parameters
import shutil

def main(params):

    if params["synth_data"]:
        synth_main(params)
        
    else:
        real_main(params)

   

if __name__=="__main__":
    import yaml, sys, shutil

    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    with open(infile, 'rb') as parfile:
        params_dict = yaml.load(parfile)

    params = setup_adjoint_contraction_parameters()
    params.update(params_dict)
    assert outfile == params["sim_file"]

    shutil.copy(infile, params["outdir"] + "/input.yml")
        


    main(params)

  
    
