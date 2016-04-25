import os
from numpy import logspace, multiply
from itertools import product
import yaml
#from campass.setup_optimization import setup_adjoint_contraction_parameters

filepath= os.path.dirname(os.path.abspath(__file__))
OUTPATH_REAL = filepath+"/results/real{}/patient_{}/alpha_{}/regpar_{}/{}"
OUTPATH_SYNTH = filepath+"/results/synthetic_noise_{}/patient_{}/alpha_{}/regpar_{}/{}"

def main():

    # Synthetic data or not
    synth_data = False
    noise = False # if synthetic data is chosen

    ### Combinations ###

    # Patient parameters

    # Which patients
    # patients = ["Impact_p8_i56", 
#                 "Impact_p9_i49", 
#                 "Impact_p10_i45", 
#                 "Impact_p12_i45", 
#                 "Impact_p14_i43", 
#                 "Impact_p15_i38"]
# #   
    patients = ["Impact_p16_i43"]
    #patients = ["CRID-pas_ESC"]
    # Optimization parameters
    
    # Weighting of strain and volume
    alphas = [0.4]
    #alphas = [i/10.0 for i in range(11)]
    #alphas = [i/100.0 for i in range(11)] + [i/10.0 for i in range(1,11)]
    # Regularization
    #reg_pars = logspace(-10,-1, 10).tolist() + multiply(5, logspace(-10, -1, 10)).tolist()
    reg_pars = [0.0]
    # Weighting of strain and volume for passive phase
    alpha_matparams = [1.0]


    ### Fixed for all runs ###
    
    # Space for contraction parameter
    gamma_space = "CG_1"
    # Use gamma from previous iteration as intial guess
    nonzero_initial_guess = False
    # Optimize material parameters or use initial ones
    optimize_matparams = True

    # Use spatial strain fields
    use_deintegrated_strains = False
    # Spring constant at base
    base_spring_k = 1.0
    # Initial material parameters
    #material_parameters = {"a":0.795, "b":6.855, "a_f":21.207, "b_f":40.545}
    material_parameters = {"a":0.291, "a_f":2.582, "b":5.0, "b_f":5.0}
    patient_type = "full"

    ### Run combinations ###
    # How to apply the weights on the strain (direction, rule, custom weights)
    # Not custom weights not implemented yet
    weights = [("all", "equal", None)]
    # Resolution of mesh
    resolutions = ["med_res"]
    # Fiber angles (endo, epi)
    fiber_angles = [(40,50)]

    # Find all the combinations
    comb = list(product(patients, resolutions, fiber_angles, weights, alphas, reg_pars, alpha_matparams))

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
        params["Patient_parameters"]["patient"] = c[0]
        params["Patient_parameters"]["resolution"] = c[1]
        params["Patient_parameters"]["fiber_angle_endo"] = c[2][0]
        params["Patient_parameters"]["fiber_angle_epi"] = c[2][1]
        params["Patient_parameters"]["weight_direction"] = c[3][0]
        params["Patient_parameters"]["weight_rule"] = c[3][1]
        params["alpha"] = c[4]
        params["reg_par"] = c[5]
        params["alpha_matparams"] = c[6]


        params["base_spring_k"] = base_spring_k
        params["optimize_matparams"] = optimize_matparams
        params["nonzero_initial_guess"] = nonzero_initial_guess
        params["use_deintegrated_strains"] = use_deintegrated_strains
        params["gamma_space"] = gamma_space
        params["Material_parameters"] = material_parameters
        params["synth_data"] = synth_data
        params["noise"] = noise
        params["Patient_parameters"]["patient_type"] = "full"


        if synth_data:
            outdir = OUTPATH_SYNTH.format(synth_data, c[0], c[4], c[5], c[1])

        else:
            scalar_str = "_scalar" if gamma_space == "R_0" else ""
            outdir = OUTPATH_REAL.format(scalar_str, c[0], c[4], c[5], c[1])

        # Make directory if it does not allready exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        params["outdir"] = outdir
        params["sim_file"] = "/".join([outdir, "result.h5"])

        # Dump paramters to yaml
        with open(fname.format(t), 'wb') as parfile:
            yaml.dump(params, parfile, default_flow_style=False)
        t += 1

    #os.system("sbatch save_patient_data.slurm {} {}".format(t0, t-1))
    os.system("sbatch run_submit.slurm {} {}".format(t0, t-1))


if __name__ == "__main__":
    main()
