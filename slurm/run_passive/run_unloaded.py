#!/usr/bin/env python

import os
from numpy import logspace, multiply
from itertools import product
import yaml

patients = ["JohnDoe"]
filepath= os.path.dirname(os.path.abspath(__file__))
OUTPATH = os.path.join(filepath,"passive/unloaded_{}/base_spring_{}/patient_{}/")

def main():

    ### Cobinations ###
    unload = True


    # Space for contraction parameter
    matparams_space = "R_0"

    ### Fixed for all runs ###
    opttargets = {"volume":True,
                  "rv_volume": False,
                  "regional_strain":True,
                  "full_strain":False,
                  "GL_strain":False,
                  "GC_strain":False,
                  "displacement":False}
    
    optweight_active = {"volume":0.95, 
                        "regional_strain": 0.05, 
                        "regularization": 0.1}
    optweight_passive = {"volume":1.0,
                         "regional_strain": 0.0}
                  
    fiber_angle_epi = -60
    fiber_angle_endo = 60
    
    
    # Optimize material parameters or use initial ones
    optimize_matparams = True

 
    # Spring constant at base
    base_spring_k = 1.0
    pericardium_spring = 0.0
    # Initial material parameters
    material_parameters = {"a":2.28, "a_f":1.685, "b":9.726, "b_f":15.779}


    ### Run combinations ###

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


    for patient in patients:

        params = {"Patient_parameters":{},
                  "Optimization_parameters":{}, 
                  "Optimization_targets":{},
                  "Unloading_parameters":{"unload_options":{}},
                  "Active_optimization_weigths": {},
                  "Passive_optimization_weigths":{},
                  "Optimization_parameters": {}}


        params["optimize_matparams"] = optimize_matparams
        params["matparams_space"] = matparams_space
        params["log_level"] = 20
        params["passive_weights"] = "-1"
        params["initial_guess"] = initial_guess
        params["Patient_parameters"]["patient"] = patient
        params["active_relax"] = 1.0
        params["base_spring_k"] = base_spring_k
        params["pericardium_spring"] = pericardium_spring
        params["passive_relax"] = 1.0
        params["unload"] = unload

        params["Optimization_parameters"]["passive_maxiter"] = 100
        params["Optimization_parameters"]["passive_opt_tol"] = 1e-10

        
        if unload:
            params["Unloading_parameters"]["maxiter"] = 10
            params["Unloading_parameters"]["tol"] = 1e-3
            params["Unloading_parameters"]["continuation"] = True
            params["Unloading_parameters"]["method"] = "fixed_point"
            params["Unloading_parameters"]["unload_options"]["maxiter"] = 50
            params["Optimization_parameters"]["passive_maxiter"] = 100


        params["Patient_parameters"]["fiber_angle_epi"] = fiber_angle_epi
        params["Patient_parameters"]["fiber_angle_endo"] = fiber_angle_endo
        
        params["Patient_parameters"]["mesh_type"] = "lv"
        params["Patient_parameters"]["mesh_group"] = ""
        
        params["Patient_parameters"]["pressure_path"] = os.path.join(filepath,"relative_path_to_pressure_data")
        params["Patient_parameters"]["mesh_path"] = os.path.join(filepath,"relative_path_to_mesh_data")
        



        for k, v in opttargets.iteritems():
            params["Optimization_targets"][k] = v


        for k, v in optweight_active.iteritems():
            params["Active_optimization_weigths"][k] = v

        for k, v in optweight_passive.iteritems():
            params["Passive_optimization_weigths"][k] = v
            

        params["Material_parameters"] = material_parameters
        params["Patient_parameters"]["patient_type"] = "full"



        outdir = OUTPATH.format(unload, base_spring_k, patient)

        # Make directory if it does not allready exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        params["outdir"] = outdir
        params["sim_file"] = "/".join([outdir, "result.h5"])

        # Dump paramters to yaml
        with open(fname.format(t), 'wb') as parfile:
            yaml.dump(params, parfile, default_flow_style=False)
        t += 1


    os.system("sbatch run_unloaded_submit.slurm {} {}".format(t0, t-1))


if __name__ == "__main__":
    main()
