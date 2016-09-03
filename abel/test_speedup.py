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
import os
from numpy import logspace, multiply
from itertools import product
import yaml


filepath= os.path.dirname(os.path.abspath(__file__))
OUTPATH = filepath+"/results/speedup/space_{}/resolution_{}/ncores_{}"

def main():

    patient = "Joakim"
 
    alpha = 0.9
    reg_par = 0.01
    alpha_matparams = 1.0
    compressibility = "incompressible"
    incompressibility_penalty = 0.0
    nonzero_initial_guess = False
    optimize_matparams = True
    use_deintegrated_strains = False
    base_spring_k = 1.0
    material_parameters = {"a":0.291, "a_f":2.582, "b":5.0, "b_f":5.0}
    patient_type = "full"
    weights = ["all", "equal", None]
    fiber_angles = [40,50]


    ncores = [1,2,4,6,8]
    gamma_spaces = ["R_0", "regional", "CG_1"]
    resolutions = ["low_res", "med_res"]

    # Find all the combinations
    comb = list(product(ncores, gamma_spaces, resolutions))

    # Directory where we dump the paramaters
    input_directory = "input"
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)

    fname = input_directory + "/file_{}.yml"
    
    # Find which number we 
    t = 1
    while os.path.exists(fname.format(t)):
        t += 1
    t0 = t


    for c in comb:

        #params = setup_adjoint_contraction_parameters()
        params = {"Patient_parameters":{}}
        params["Patient_parameters"]["patient"] = patient
        params["Patient_parameters"]["resolution"] = c[2]
        params["Patient_parameters"]["fiber_angle_endo"] = fiber_angles[0]
        params["Patient_parameters"]["fiber_angle_epi"] = fiber_angles[1]
        params["Patient_parameters"]["weight_direction"] = weights[0]
        params["Patient_parameters"]["weight_rule"] = weights[1]
        params["alpha"] = alpha
        params["reg_par"] = reg_par
        params["alpha_matparams"] = alpha_matparams


        params["compressibility"] = compressibility
        params["incompressibility_penalty"] = incompressibility_penalty
        params["base_spring_k"] = base_spring_k
        params["optimize_matparams"] = optimize_matparams
        params["nonzero_initial_guess"] = nonzero_initial_guess
        params["use_deintegrated_strains"] = use_deintegrated_strains
        params["gamma_space"] = c[1]
        params["Material_parameters"] = material_parameters
        params["synth_data"] = False
        params["noise"] = False
        params["Patient_parameters"]["patient_type"] = "full"

        outdir = OUTPATH.format(c[1], c[2], c[0])

        # Make directory if it does not allready exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        params["outdir"] = outdir
        params["sim_file"] = "/".join([outdir, "result.h5"])

        # Dump paramters to yaml
        with open(fname.format(t), 'wb') as parfile:
            yaml.dump(params, parfile, default_flow_style=False)
        

        os.system("sbatch test_speedup.slurm {} {} {}".format(c[0], outdir, t))
        t += 1
        

if __name__ == "__main__":
    main()
