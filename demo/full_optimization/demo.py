#!/usr/bin/env python
import os, yaml
from pulse_adjoint.run_full_optimization import main
from pulse_adjoint.setup_optimization import setup_adjoint_contraction_parameters, setup_material_parameters

#Path to this file
filepath= os.path.dirname(os.path.abspath(__file__))
# Path to results
OUTPATH = os.path.join(filepath,"results/{}")



patient = "simple_ellipsoid"
# patient = "prolate_ellipsoid"
# patient = "lbbb"


# Some containers for the parameters to be parsed to the program
params = {"Patient_parameters":{},
          "Optimization_parameters":{}, 
          "Optimization_targets":{},
          "Unloading_parameters":{"unload_options":{}},
          "Active_optimization_weigths": {},
          "Passive_optimization_weigths":{},
          "Optimization_parameters": {}}
opttargets = {"volume":True,
              "regional_strain":True}
for k, v in opttargets.iteritems():
    params["Optimization_targets"][k] = v
    


    
######################
# Passive optimization
######################

params["material_model"] = "holzapfel_ogden"
# params["material_model"] = "guccione"
# params["material_model"] = "neo_hookean"
# Initial material parameters
params["Material_parameters"] = setup_material_parameters(params["material_model"])

# Optimize matierial parameters or just use the initial guess
params["optimize_matparams"] = False

# Space for material paramter R_0 refers to scalar
params["matparams_space"] = "R_0"

# Optimize only end-diatolic point.
# To optimize all point set this to 'all'
params["passive_weights"] = "all"

# Estimate the unloaded geometry
params["unload"] = False

if params["unload"]:
    params["Unloading_parameters"]["maxiter"] = 10
    params["Unloading_parameters"]["tol"] = 1e-4
    params["Unloading_parameters"]["continuation"] = True
    params["Unloading_parameters"]["method"] = "fixed_point"
    params["Unloading_parameters"]["estimate_initial_guess"] = True
    params["Unloading_parameters"]["unload_options"]["maxiter"] = 15
    params["Unloading_parameters"]["unload_options"]["tol"] = 1e-8
    params["Optimization_parameters"]["passive_maxiter"] = 10
else:
    params["Optimization_parameters"]["passive_maxiter"] = 100


# Weights on the functional (volume only)
optweight_passive = {"volume":1.0,
                     "regional_strain": 0.0}

for k, v in optweight_passive.iteritems():
    params["Passive_optimization_weigths"][k] = v
            



######################
# Active optimization
######################

# Space for active contraction parameter
# params["gamma_space"] = "CG_1"
params["gamma_space"] = "R_0"

# Active model
# params["active_model"]  = "active_strain"
params["active_model"]  = "active_stress"


# Use gamma from previous iteration as intial guess
params["initial_guess"] ="previous"


# We set the upper bound for the contraction parameter to 1.0
# in order to not scale the functional gradient
params["Optimization_parameters"]["gamma_max"] = 1.0

if params["active_model"] == "active_strain":
    
    params["T_ref"] = 0.5

else: 
    params["T_ref"] = 75.0
    params["eta"] = 0.2

# Weights on the functional
optweight_active = {"volume":0.95, 
                    "regional_strain": 0.05, 
                    "regularization": 0.1}

for k, v in optweight_active.iteritems():
    params["Active_optimization_weigths"][k] = v
    

###############
# Other options
###############

# Log level (10=debug, 20=info)
params["log_level"] = 20

# Stiffness of Robin type spring on the base
params["base_spring_k"] = 10.0
    
            
####################
# Patient paramteres
####################

# Name of the patient 
params["Patient_parameters"]["patient"] = patient

# This is just some folder within the file containing the mesh
if patient == "lbbb":
    params["Patient_parameters"]["mesh_group"] = "41"
elif patient == "simple":
    params["Patient_parameters"]["mesh_group"] = ""


# Path to pressure data
params["Patient_parameters"]["pressure_path"] \
    = os.path.join(filepath, "data/measurements_{}.yml".format(patient))

params["Patient_parameters"]["mesh_path"] \
    = os.path.join(filepath,"data/geometry_{}.h5".format(patient))



##############
# Run the code
##############

# Path to results
outdir = OUTPATH.format(patient)
params["sim_file"] = "/".join([outdir, "result.h5"])

# Make directory if it does not allready exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Delete the old results file
if os.path.isfile(params["sim_file"]):
    os.remove(params["sim_file"])
    
# Load default parameters. Have a look at this dictionary for additional options
params_ = setup_adjoint_contraction_parameters()
# Update parameters accordingly
params_.update(params)

# Save parameters for later processing
with open("/".join([outdir, "input.yml"]), "wb") as f:
    yaml.dump(params_.to_dict(), f, default_flow_style=False)


# For the synthtic case we need to generate some data
# You don't have to do this every tim
if patient in ["simple_ellipsoid","prolate_ellipsoid"]:
    from generate_data import generate_synthetic_data
    from dolfin import MPI, mpi_comm_world
    comm = mpi_comm_world()
    if MPI.size(comm) == 1:
        generate_synthetic_data(patient, **params)

    else:
        if not os.path.isfile(params["Patient_parameters"]["pressure_path"]):
            if MPI.rank(comm) == 0:
                print("Data does not exist. Try running the script in serial first")

    
main(params_)

