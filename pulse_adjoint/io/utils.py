#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
from .io_import import *

def passive_inflation_exists(params):
    

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


if has_h5py:
    parallel_h5py = h5py.h5.get_config().mpi
else:
    parallel_h5py = False
    
def gather_dictionary(d):

    def gather_dict(a):
        v = {}
        for key, val in a.iteritems():
            
            if isinstance(val, dict):
                v[key] = gather_dict(val)

            elif isinstance(val, (list,  np.ndarray, tuple)):
                                        
                    if len(val) == 0:
                        # If the list is empty we do nothing
                        pass
                    
                    elif isinstance(val[0], (dolfin.Vector, dolfin.GenericVector)):
                        v[key] = {}
                        for i, f in enumerate(val):
                            v[key][i] = gather_broadcast(f.array())
                            
                                                      
                    elif isinstance(val[0], (dolfin.Function, dolfin_adjoint.Function)):
                        v[key] = {}
                        for i, f in enumerate(val):
                            v[key][i] = gather_broadcast(f.vector().array())
                                                                         
                            
                    elif isinstance(val[0], (float, int)):
                        v[key] = np.array(val, dtype=float)

                        
                    elif isinstance(val[0], list) or isinstance(val[0], np.ndarray) \
                         or  isinstance(val[0], dict):
                        # Make this list of lists into a dictionary
                        f = {str(i):v for i,v in enumerate(val)}
                        v[key] = gather_dict(f)                
                    
                    else:
                        raise ValueError("Unknown type {}".format(type(val[0])))
                    
            elif isinstance(val, (float, int)):
                v[key] = np.array([float(val)], dtype=float)

    
            elif isinstance(val, (dolfin.Vector, dolfin.GenericVector)):
                v[key] = gather_broadcast(val.array())
                
                
            elif isinstance(val, (dolfin.Function, dolfin_adjoint.Function)):
                v[key] = gather_broadcast(val.vector().array())
                
            else:
                raise ValueError("Unknown type {}".format(type(val)))

        return v

    return gather_dict(d)
            
            

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
                    

    # First gather the whole dictionary as numpy arrays


    
        
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



def numpy_dict_to_h5(d, h5name, h5group = "", comm = dolfin.mpi_comm_world(),
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
                    
    if comm.rank == 0:
        with h5py.File(h5name, file_mode) as h5file:

            def dict2h5(a, group):
                
                for key, val in a.iteritems():
                
                    subgroup = "/".join([group, str(key)])
                    
                    if isinstance(val, dict):
                        dict2h5(val, subgroup)

                    else:
                        assert isinstance(val, np.ndarray)
                        assert val.dtype == np.float
                    
                        h5file.create_dataset(subgroup,data= val)

            dict2h5(d, h5group)

    MPI.barrier(comm)
    
        
