from pscm.run_full_optimization import main as real_main
from pscm.synthetic_data import run_active_synth_data as synth_main
from pscm.adjoint_contraction_args import *
from pscm.setup_optimization import setup_adjoint_contraction_parameters
from numpy_mpi import mpi_print



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

    from IPython import embed; embed()

    shutil.copy(infile, params["outdir"] + "/input.yml")
        


    main(params)

  
    
