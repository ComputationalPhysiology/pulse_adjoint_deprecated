#!/usr/bin/env python
# Copyright (C) 2016 Henrik Finsberg
#
# This file is part of PULSE-ADJOINT.
#
# PULSE-ADJOINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PULSE-ADJOINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PULSE-ADJOINT. If not, see <http://www.gnu.org/licenses/>.
from dolfinimport import *
import numpy as np
import h5py, yaml, os
from adjoint_contraction_args import *
from numpy_mpi import *



def write_opt_results_to_h5(h5group,
                            params,
                            for_result_opt,
                            solver,
                            opt_result):

    filename = params["sim_file"]
    filedir = os.path.abspath(os.path.dirname(params["sim_file"]))
    if not os.path.exists(filedir) and mpi_comm_world().rank == 0:
        os.makedirs(filedir)
        write_append = "w"

    if os.path.isfile(filename):
        # Open the file in h5py
        h5file_h5py = h5py.File(filename, 'a')
        # Check if the group allready exists
        if h5group in h5file_h5py:
            # If it does, delete it
            if mpi_comm_world().rank == 0:
                del h5file_h5py[h5group]
        # Close the file
        h5file_h5py.close()
        # Open the file in HDF5File format
        open_file_format = "a"
        
    else:
        open_file_format = "w"
        
    
    
    with HDF5File(mpi_comm_world(), filename, open_file_format) as h5file:

        def save_data(data, name):
            # Need to do this in order to not get duplicates in parallell
            if hasattr(data, "__len__"):
                # Size of mesh needs to be big enough so that it can be distrbuted
                f = Function(VectorFunctionSpace(UnitSquareMesh(100,100), "R", 0, dim = len(data)))
                
            else:
                f = Function(FunctionSpace(UnitSquareMesh(100,100), "R", 0))
                

            f.assign(Constant(data))
            h5file.write(f.vector(), name)

        
        # Dump parameters to yaml file as well
        with open(filedir+"/parameters.yml", 'w') as parfile:
            yaml.dump(params.to_dict(), parfile, default_flow_style=False)

        # Optimization results
        if opt_result is not None:
            for k, v in opt_result.iteritems():

                # This is a list of vectors
                if k == "controls":
                    for it, c in enumerate(v):
                        h5file.write(c, h5group + \
                                     "/controls/{}".format(it))
                else:
                    # Do not save empty lists
                    if (hasattr(v, "__len__") and not len(v) == 0) or np.isscalar(v):
                        save_data(v, "/".join([h5group, k]))
                  
            # controls = opt_result.pop("controls", [0])
            # for it, c in enumerate(controls):
            #     h5file.write(c, h5group + "/controls/{}".format(it))

            # func_vals= np.array(opt_result.pop("func_vals", [0]))
            # save_data(func_vals, h5group + "/funtion_values")

            # for_times= np.array(opt_result.pop("forward_times", [0]))
            # save_data(for_times, h5group + "/forward_times")

            # back_times= np.array(opt_result.pop("backward_times", [0]))
            # save_data(back_times, h5group + "/backward_times")

            
            
            # if opt_result and isinstance(opt_result, dict):
            #     dump_parameters_to_attributes(opt_result, h5group)
                

        # States
        for i, w in enumerate(for_result_opt["states"]):
            assign_to_vector(solver.get_state().vector(), gather_broadcast(w.array()))
            h5file.write(solver.get_state(), "/".join([h5group, "states/{}".format(i)]))

            u,p = solver.get_state().split(deepcopy=True)
            h5file.write(u, "/".join([h5group, "displacement/{}".format(i)]))
            h5file.write(p, "/".join([h5group, "lagrange_multiplier/{}".format(i)]))

        # Control
        # h5file.write(for_result_opt["initial_control"],
        #               "/".join([h5group, "initial_control"]))
        h5file.write(for_result_opt["optimal_control"],
                      "/".join([h5group, "optimal_control"]))
        

        # Store the optimization targets:
        for k, v in for_result_opt["optimization_targets"].iteritems():
            group = "/".join([h5group, "optimization_targets", k])

            # Save the results
            save_data(np.array(v.results["func_value"]), "/".join([group, "func_value"]))

            # Save the optimal value
            save_data(np.array(v.results["func_value"]), "/".join([group, "func_value"]))
            
            # Save weights if applicable
            if hasattr(v, 'weights'):
                for l,w in enumerate(v.weights):
                    h5file.write(w.vector(), "/".join([group, "weigths", str(l)]))
                
                

            for k1 in ["simulated", "target"]:
                n = len(v.results[k1])
                for i in range(n):
                    if k == "regional_strain":
                        for j,s in enumerate(v.results[k1][i]):
                            h5file.write(s, "/".join([group, k1, str(i), "region_{}".format(j)]))
                        
                    else:
                        h5file.write(v.results[k1][i], "/".join([group, k1, str(i)]))
                    
            
        # Store the regularization
        if for_result_opt.has_key("regularization"):
            save_data(for_result_opt["regularization"].results["func_value"],
                      "/".join([h5group, "regularization", "func_value"]))
                                                      
        

        # Save boundary conditions
        for k,v in for_result_opt["bcs"].iteritems():
            save_data(np.array(v), "/".join([h5group, "bcs", k]))

