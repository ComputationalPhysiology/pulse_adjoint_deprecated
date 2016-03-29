import os
from itertools import product
import yaml
from pscm.setup_optimization import setup_adjoint_contraction_parameters

filepath= os.path.dirname(os.path.abspath(__file__))
OUTPATH_REAL = filepath+"/results/real/patient_{}/alpha_{}/regpar{}/rule_{}_dir_{}/{}"
OUTPATH_SYNTH = filepath+"/results/synthetic_noise_{}/patient_{}/alpha_{}/regpar{}/rule_{}_dir_{}/{}"

def main():

    ### Combinations ###

    # Patient parameters

    # Which patients
    patients = ["Impact_p16_i43"]
    # How to apply the weights on the strain (direction, rule, custom weights)
    # Not custom weights not implemented yet
    weights = [("all", "equal", None)]
    # Resolution of mesh
    resolutions = ["low_res"]
    # Fiber angles (endo, epi)
    fiber_angles = [(40,50)]


    # Optimization parameters
    
    # Weighting of strain and volume
    alphas = [0.5]
    # Regularization
    reg_pars = [0.1]
    # Weighting of strain and volume for passive phase
    alpha_matparams = [1.0]


    ### Fixed for all runs ###
    
    # Synthetic data or not
    synth_data = False
    noise = False # if synthetic data is chosen
    patient_type = "full"

    # Initial material parameters
    material_parameters = {"a":0.795, "b":6.855, "a_f":21.207, "b_f":40.545}
    # Space for contraction parameter
    gamma_space = "CG_1"
    # Use spatial strain fields
    use_deintegrated_strains = False
    # Optimize material parameters or use initial ones
    optimize_matparams = True
    # Spring constant at base
    base_spring_k = 10.0


    ### Run combinations ###

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

        params = setup_adjoint_contraction_parameters()

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
        params["use_deintegrated_strains"] = use_deintegrated_strains
        params["gamma_space"] = gamma_space
        params["Material_parameters"].update(material_parameters)
        params["synth_data"] = synth_data
        params["noise"] = noise
        params["Patient_parameters"]["patient_type"] = "full"


        if synth_data:
            outdir = OUTPATH_SYNTH.format(synth_data, c[0], c[4], 
                                          c[5], c[3][1], c[3][0], c[1])

        else:
            outdir = OUTPATH_REAL.format(c[0], c[4], c[5], c[3][1], 
                                         c[3][0], c[1])

        # Make directory if it does not allready exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        params["outdir"] = outdir
        params["sim_file"] = "/".join([outdir, "result.h5"])

        # Dump paramters to yaml
        with open(fname.format(t), 'wb') as parfile:
            yaml.dump(params.to_dict(), parfile, default_flow_style=False)
        t += 1

    os.system("sbatch run_submit.slurm {} {}".format(t0, t-1))


if __name__ == "__main__":
    main()
