"""
If fiber fields that are not saved togther with the mesh
is used. You shoul run the script in serial first
in order to generate the fiber fields and save them.

Then the fibers can be loaded in parallell.

"""

from campass.setup_optimization import initialize_patient_data, setup_general_parameters, setup_adjoint_contraction_parameters


def main(params):

    setup_general_parameters()

    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])

    


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
