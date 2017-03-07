import h5py, os, mpi4py, petsc4py
import numpy as np
import dolfin, dolfin_adjoint

from .utils import Text
from .adjoint_contraction_args import logger
from .numpy_mpi import *

parallel_h5py = h5py.h5.get_config().mpi


def open_h5py(h5name, file_mode="a", comm= dolfin.mpi_comm_world()):

    assert isinstance(comm, (petsc4py.PETSc.Comm, mpi4py.MPI.Intracomm))
    if parallel_h5py:
        if isinstance(comm, petsc4py.PETSc.Comm):
            comm = comm.tompi4py()
        
        return  h5py.File(h5name, file_mode, driver='mpio', comm=comm)
    else:
        return  h5py.File(h5name, file_mode)
    
def check_and_delete(h5name, h5group, comm= dolfin.mpi_comm_world()):

    with open_h5py(h5name, "a", comm) as h5file:
        if h5group in h5file:

            if parallel_h5py:
                
                logger.debug("Deleting existing group: '{}'".format(h5group))
                del h5file[h5group]

            else:
                if comm.rank == 0:

                    logger.debug("Deleting existing group: '{}'".format(h5group))
                    del h5file[h5group]
        
            



def dict2h5_hpc(d, h5name, h5group = "", comm = dolfin.mpi_comm_world(),
                overwrite_file = True, overwrite_group=True):
    """Create a HDF5 file and put the
    data in the dictionary in the 
    same hiearcy in the HDF5 file
    
    Assume leaf of dictionary is either
    float, numpy.ndrray, list or 
    dolfin.GenericVector.

    :param d: Dictionary to be saved
    :param h5fname: Name of the file where you want to save
    
    
    """
    if overwrite_file:
        if os.path.isfile(h5name):
            os.remove(h5name)

    
    file_mode = "a" if os.path.isfile(h5name) and not overwrite_file else "w"

    # IF we should append the file but overwrite the group we need to
    # check that the group does not exist. If so we need to open it in
    # h5py and delete it.
    if file_mode == "a" and overwrite_group and h5group!="":
        check_and_delete(h5name, h5group, comm)
                    
    
    with open_h5py(h5name, file_mode, comm) as h5file:

        def dict2h5(a, group):
            
            for key, val in a.iteritems():
               
                subgroup = "/".join([group, str(key)])
                    
                if isinstance(val, dict):
                    dict2h5(val, subgroup)
                    
                elif isinstance(val, (list,  np.ndarray, tuple)):
                                        
                    if len(val) == 0:
                        # If the list is empty we do nothing
                        pass
                    
                    elif isinstance(val[0], (dolfin.Vector, dolfin.GenericVector)):
                        for i, f in enumerate(val):

                            if parallel_h5py:
                                v = f.array()
                                h5file.create_dataset(subgroup + "/{}".format(i),data = v)
                            else:
                                v = gather_broadcast(f.array())
                                
                                if comm.rank == 0:
                                    h5file.create_dataset(subgroup + "/{}".format(i),data = v)
                                         
                            
                    elif isinstance(val[0], (dolfin.Function, dolfin_adjoint.Function)):
                        for i, f in enumerate(val):
                            
                            if parallel_h5py:
                                v = f.vector().array()
                                h5file.create_dataset(subgroup + "/{}".format(i),data=v)
                            else:
                                v = gather_broadcast(f.vector().array())
                                if comm.rank == 0:
                                    h5file.create_dataset(subgroup + "/{}".format(i),data=v)
                            
                                                      
                                         
                            
                    elif isinstance(val[0], (float, int)):
                       
                        v = np.array(val, dtype=float)
                        if parallel_h5py:
                            h5file.create_dataset(subgroup, data=v)
                        else:
                            if comm.rank == 0:
                                h5file.create_dataset(subgroup, data=v)
                            
                        
                    elif isinstance(val[0], list) or isinstance(val[0], np.ndarray) \
                         or  isinstance(val[0], dict):
                        # Make this list of lists into a dictionary
                        f = {str(i):v for i,v in enumerate(val)}
                        dict2h5(f, subgroup)                
                    
                    else:
                        raise ValueError("Unknown type {}".format(type(val[0])))
                    
                elif isinstance(val, (float, int)):
                    v = np.array([float(val)], dtype=float)
                    
                    if parallel_h5py:
                        h5file.create_dataset(subgroup, data = v)
                    else:
                        if comm.rank == 0:
                            h5file.create_dataset(subgroup, data = v)
    
                elif isinstance(val, (dolfin.Vector, dolfin.GenericVector)):
                    
                    if parallel_h5py:
                        v = val.array()
                        h5file.create_dataset(subgroup, data = v)
                    else:
                        v = gather_broadcast(val.array())
                        if comm.rank == 0:
                            h5file.create_dataset(subgroup, data = v)
                    

                elif isinstance(val, (dolfin.Function, dolfin_adjoint.Function)):
                    
                    if parallel_h5py:
                        v = val.vector().array()
                        h5file.create_dataset(subgroup,data= v)
                    else:
                        v= gather_broadcast(val.vector().array())
                        if comm.rank == 0:
                            h5file.create_dataset(subgroup,data= v)
                    
                else:
                    raise ValueError("Unknown type {}".format(type(val)))

        dict2h5(d, h5group)
        comm.Barrier()
        


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
       
            assign_to_vector(solver.get_state().vector(), gather_broadcast(w.array()))
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
            
    
    dict2h5_hpc(data, h5name, h5group, comm, 
                overwrite_file = False, overwrite_group = False)
    

def test_store():


    from mesh_generation.mesh_utils import load_geometry_from_h5
    from setup_optimization import make_solver_params, setup_adjoint_contraction_parameters, setup_general_parameters
    from forward_runner import PassiveForwardRunner
    from run_optimization import load_targets
    

    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    
    geo = load_geometry_from_h5("../demo/data/mesh.h5", "22")
    
    measurements = yaml.load(open("../demo/data/measurements.yml"))
    measurements.pop("original_strain")
    strain = measurements.pop("strain")
    measurements["regional_strain"] = strain
    measurements["pressure"] =  np.subtract(measurements["pressure"],
                                            measurements["pressure"][0])
    
    
    solver_parameters, pressure, control = make_solver_params(params, geo)
    
    optimization_targets, bcs = load_targets(params, solver_parameters, measurements)
    
    for_run = PassiveForwardRunner(solver_parameters, 
                                   pressure, 
                                   bcs,
                                   optimization_targets,
                                   params, 
                                   control)


    # from IPython import embed; embed()
    # exit()
    for_run.assign_material_parameters(control)
    
    phm, w= for_run.get_phm(False, True)
    
    functional = for_run.make_functional()
    for_run.update_targets(0, split(w)[0], control)
    for_run.states = [Vector(w.vector())]
    for_res = for_run._make_forward_result([0.0], [functional*dt[0.0]])

    for_res["initial_control"] = Vector(control.vector())
    for_res["optimal_control"] = control

    opt_result = {}
    opt_result["nfev"] = 3
    opt_result["nit"] = 3
    opt_result["njev"] = 3
    opt_result["ncrash"] = 1
    opt_result["run_time"] = 321.32
    opt_result["controls"] = [control]
    opt_result["func_vals"] = [0.4]
    opt_result["forward_times"] = [123.2]
    opt_result["backward_times"] = [214.2]
    opt_result["grad_norm"] = [0.024]
    


    params["sim_file"] = "test.h5"
    h5group = "active"

    write_opt_results_to_h5(h5group,
                            params,
                            for_res,
                            phm.solver,
                            opt_result)
    
    

def passive_inflation_exists(params):
    
    from adjoint_contraction_args import PASSIVE_INFLATION_GROUP

    if not os.path.exists(params["sim_file"]):
        return False
    
    h5file = open_h5py(params["sim_file"], "r")
    key = PASSIVE_INFLATION_GROUP

    # Check if pv point is already computed
    if key in h5file.keys():
        logger.info(Text.green("Passive inflation, {}".format("fetched from database")))
        h5file.close()
        return True
    logger.info(Text.blue("Passive inflation, {}".format("Run Optimization")))
    h5file.close()
    return False

def contract_point_exists(params):
    
    from adjoint_contraction_args import ACTIVE_CONTRACTION, CONTRACTION_POINT, PASSIVE_INFLATION_GROUP, PHASES
    
    if not os.path.exists(params["sim_file"]):
        logger.info(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")
        return False

    h5file = open_h5py(params["sim_file"], "r")
    key1 = ACTIVE_CONTRACTION
    key2  = CONTRACTION_POINT.format(params["active_contraction_iteration_number"])
    key3 = PASSIVE_INFLATION_GROUP
    
	
    if not key3 in h5file.keys():
        logger.info(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")

    
    if params["phase"] == PHASES[0]:
        h5file.close()
        return False
    
    if not key1 in h5file.keys():
        h5file.close()
        return False

    try:
        
        # Check if pv point is already computed
        if key1 in h5file.keys() and key2 in h5file[key1].keys():
            pressure = np.array(h5file[key1][key2]["bcs"]["pressure"])[-1]
            logger.info(Text.green("Contract point {}, pressure = {:.3f} {}".format(params["active_contraction_iteration_number"],
                                                                                    pressure, "fetched from database")))
            h5file.close()
            return True
        logger.info(Text.blue("Contract point {}, {}".format(params["active_contraction_iteration_number"],"Run Optimization")))
        h5file.close()
        return False
    except KeyError:
        h5file.close()
        return False
