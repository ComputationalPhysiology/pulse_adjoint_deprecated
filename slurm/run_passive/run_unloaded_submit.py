#!/usr/bin/env python
from pulse_adjoint.run_full_optimization import main as real_main
from pulse_adjoint.adjoint_contraction_args import *
from pulse_adjoint.setup_optimization import setup_adjoint_contraction_parameters
import shutil


def main():
    import yaml, sys, shutil, os

    infile = sys.argv[1]
    outfile = sys.argv[2]
    keep = len(sys.argv) == 4
    with open(infile, 'rb') as parfile:
        params_dict = yaml.load(parfile)
        
    params = setup_adjoint_contraction_parameters()
    params.update(params_dict)
    assert outfile == params["sim_file"]
            
    if not keep:
        params["outdir"] = "."
        params["sim_file"] = os.path.basename(params["sim_file"])
        # Assume mesh is copied to current directory
        params["Patient_parameters"]["mesh_path"] = os.path.basename(params["Patient_parameters"]["mesh_path"])
        params["Patient_parameters"]["pressure_path"] = os.path.basename(params["Patient_parameters"]["pressure_path"])

    
    shutil.copy(infile, params["outdir"] + "/input.yml")
    real_main(params, passive_only = True)
    


if __name__=="__main__":
    main()
