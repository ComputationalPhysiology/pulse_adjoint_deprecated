from .io_import import *
from .utils import *

def write_opt_results_to_h5(h5group,
                            params,
                            for_result_opt,
                            solver,
                            opt_result,
                            comm = dolfin.mpi_comm_world()):

    
    h5name = params["sim_file"]
    logger.info("Save results to {}:{}".format(h5name, h5group))

    filedir = os.path.abspath(os.path.dirname(params["sim_file"]))
    if not os.path.exists(filedir) and comm.rank == 0:
        os.makedirs(filedir)
        write_append = "w"


    if os.path.isfile(h5name):
        # Open the file in h5py
        check_and_delete(h5name, h5group, comm)
        # Open the file in HDF5File format
        open_file_format = "a"
        
    else:
        open_file_format = "w"
        

    # Make sure to save the state as a function, and
    # make sure that we don't destroy the dof-structure
    # by first assigning the state to te correction function
    # and then save it.
    with dolfin.HDF5File(comm, h5name, open_file_format) as h5file:

       
        h5file.write(for_result_opt["optimal_control"],
                     "/".join([h5group, "optimal_control"]))
        
       
        # States
        for i, w in enumerate(for_result_opt["states"]):

            solver.get_state().vector().zero()
            solver.get_state().vector().axpy(1.0, w.vector())
            h5file.write(solver.get_state(), "/".join([h5group, "states/{}".format(i)]))
       
            u,p = solver.get_state().split(deepcopy=True)
            h5file.write(u, "/".join([h5group, "displacement/{}".format(i)]))
            h5file.write(p, "/".join([h5group, "lagrange_multiplier/{}".format(i)]))


   
    data = {"initial_control":for_result_opt["initial_control"],
            "bcs":for_result_opt["bcs"],
            "optimization_results": opt_result}
    
    if  for_result_opt.has_key("regularization"):
        data["regularization"] = for_result_opt["regularization"].results

    for k,v in for_result_opt["optimization_targets"].iteritems():

        data[k] = v.results
        
        if hasattr(v, 'weights_arr'):
            data[k]["weights"] = v.weights_arr

    
    gathered_data = gather_dictionary(data)
    numpy_dict_to_h5(gathered_data, h5name, h5group, comm, 
                     overwrite_file = False, overwrite_group = False)
   
    # dict2h5_hpc(data, h5name, h5group, comm, 
    #             overwrite_file = False, overwrite_group = False)
    

    



if __name__ == "__main__":
    test_store()
    
