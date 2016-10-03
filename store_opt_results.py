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
                            ini_for_res,
                            for_result_opt, 
                            opt_matparams = None,
                            opt_gamma = None,
                            opt_result = None):

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

        def dump_parameters_to_attributes(params, group):

            for k,v in params.iteritems():

                if isinstance(v, Parameters) or isinstance(v, dict):
                    for k_sub, v_sub in v.iteritems():
                        h5file.attributes(group)["{}/{}".format(k,k_sub)] = v_sub

                else:
                    if isinstance(v, np.bool_): v = bool(v)
                    if isinstance(v, int): v = abs(v)

                    h5file.attributes(group)[k] = v
                    

        # Parameters
        if opt_matparams:
            group = "/".join([h5group, "parameters"])
            
            h5file.write(opt_matparams.vector(),
                         "/".join([group, "optimal_material_parameters"]))
            
            save_data(params["Material_parameters"].values(),
                      "/".join([group, "initial_material_parameters"]))

            
            save_data(0.0, "/".join([group, "activation_parameter"]))
            

        if opt_gamma:
            group = "/".join([h5group, "parameters"])
            
            h5file.write(opt_gamma.vector(), "/".join([group, "activation_parameter"]))
           
                



        # Input parameters
        h5file.attributes(h5group + "/parameters")["material parameters"] = \
          "a, b, a_f, b_f in transversely isotropic Holzapfel and Ogden model"
        h5file.attributes(h5group + "/parameters")["activation parameter"] = \
          "Active contraction in fiber direction. Value between 0 and 1 where 0 (starting value) is no contraction and 1 (infeasable) is full contraction"
        dump_parameters_to_attributes(params, h5group)
        
        # Dump parameters to yaml file as well
        with open(filedir+"/parameters.yml", 'w') as parfile:
            yaml.dump(params.to_dict(), parfile, default_flow_style=False)

        # Optimization results
        if opt_result is not None:
            controls = opt_result.pop("controls", [0])
            for it, c in enumerate(controls):
                h5file.write(c, h5group + "/controls/{}".format(it))

            func_vals= np.array(opt_result.pop("func_vals", [0]))
            save_data(func_vals, h5group + "/funtion_values")
            
            if opt_result and isinstance(opt_result, dict):
                dump_parameters_to_attributes(opt_result, h5group)
                

        # States
        for i, w in enumerate(for_result_opt["states"]):
            h5file.write(w, "/".join([h5group, "states/{}".format(i)]))


        # Store the optimization targets:
        for k, v in for_result_opt["optimization_targets"].iteritems():
            group = "/".join([h5group, "optimization_targets", k])

            # Save the results
            save_data(np.array(v.results["func_value"]), "/".join([group, "func_value"]))
            
            # Save weights if applicable
            if hasattr(v, 'weights'):
                for l,w in enumerate(v.weights):
                    h5file.write(w.vector(), "/".join([group, "weigths", str(l)]))
                
                
            
            n = len(v.results["func_value"])
            for k1 in ["simulated", "target"]:

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




###################################
#Test area
###################################

def make_args():
    args = lambda : None
    args.alpha = 0.5
    args.base_spring_k = 1.0
    args.opt_tol = 1.0e-8
    args.scale = 1.0
    args.num_points = 1
    args.matparams = [1.0 , 2.0 ,3.0 ,4.0]
    args.filename = "test_dummy_data.h5"
    args.mesh = UnitSquareMesh(2,2)
    args.W = FunctionSpace(args.mesh, "CG", 2)
    return args

def test_systolic_output():
    args = make_args()
    alpha = args.alpha
    gamma = 0.1
    active_pv_point_number = 0
    w = interpolate(Constant(3.14159), args.W)
    write_scalar_gamma_contraction_result_to_h5(args.filename, active_pv_point_number, alpha, gamma, w) 

def test_diastolic_output():
    args = make_args()

    for_result_opt = lambda: None
    for_result_opt.phm = lambda: None
    for_result_opt.phm.solver = lambda: None
    for_result_opt.phm.solver.w = Function(args.W)
    for_result_opt.phm.mesh = args.mesh
    
    for_result_opt.func_value_strain = 2.1
    for_result_opt.func_value_volume = 3.1
    
    #for_result_opt.states = [interpolate(Constant(1.0), W), interpolate(Constant(2.0), W)]
    v1 = Vector(mpi_comm_world(), args.W.dim())
    v2 = Vector(mpi_comm_world(), args.W.dim())
    v1[0] = 0.0
    v2[0] = 1.0
    
    for_result_opt.states = [v1, v2]
    for_result_opt.lv_pressures = [1.0, 2.0]
    for_result_opt.rv_pressures = [2.0, 3.0]
    for_result_opt.volumes = [3.0, 4.0]
    
    for_result_opt.weighted_func_value_strain = 10.1
    for_result_opt.weighted_func_value_volume = 13.1

    ini_for_res =  lambda:None

    ini_for_res.func_value_strain = 1.1
    ini_for_res.func_value_volume = 2.1
    
    ini_for_res.weighted_func_value_strain = 8.1
    ini_for_res.weighted_func_value_volume = 9.1

    opt_matparams = interpolate(Constant([4,5,6,7]), VectorFunctionSpace(args.mesh, "R", 0, dim=4))

    h5group = PASSIVE_INFLATION_GROUP.format(args.alpha)

    write_opt_results_to_h5(args.filename, h5group, args, ini_for_res, for_result_opt, opt_matparams)

def test_multiple_points_output():
    args = make_args()

    for_result_opt = lambda: None
    for_result_opt.phm = lambda: None
    for_result_opt.phm.solver = lambda: None
    for_result_opt.phm.solver.w = Function(args.W)
    for_result_opt.phm.mesh = args.mesh
    
    for_result_opt.func_value_strain = 2.1
    for_result_opt.func_value_volume = 3.1
    
    #for_result_opt.states = [interpolate(Constant(1.0), W), interpolate(Constant(2.0), W)]
    v1 = Vector(mpi_comm_world(), args.W.dim())
    v2 = Vector(mpi_comm_world(), args.W.dim())
    v1[0] = 0.0
    v2[0] = 1.0
   
    
    for_result_opt.states = [v1, v2]
    for_result_opt.lv_pressures = [1.0, 2.0]
    for_result_opt.rv_pressures = [2.0, 3.0]
    for_result_opt.volumes = [3.0, 4.0]
    
    for_result_opt.weighted_func_value_strain = 10.1
    for_result_opt.weighted_func_value_volume = 13.1

    for_result_opt.reg_par = 1.0
    for_result_opt.gamma_gradient = 0.1 
    for_result_opt.strain_weights = np.ones((17,3)).tolist()

    # for_result_opt.previous_contracting_points = zip(range(4), [range(18) for i in range(4)], range(20, 24))

    ini_for_res =  lambda:None

    ini_for_res.func_value_strain = 1.1
    ini_for_res.func_value_volume = 2.1
    
    ini_for_res.weighted_func_value_strain = 8.1
    ini_for_res.weighted_func_value_volume = 9.1

    opt_gamma = Function(VectorFunctionSpace(args.mesh, "R", 0, dim = 18))
    opt_gamma.assign(Constant([0.01]*18))
    for_result_opt.num_gamma_steps = 4.0
    contraction_point = 0
    h5group = ACTIVE_CONTRACTION_GROUP.format(args.alpha, contraction_point) + "/regional"

    write_opt_results_to_h5(args.filename, h5group, args, ini_for_res, for_result_opt, opt_gamma = opt_gamma)

    opt_gamma = Function(VectorFunctionSpace(args.mesh, "R", 0, dim = 18))
    opt_gamma.assign(Constant([0.02]*18))
    for_result_opt.num_gamma_steps = 6.0
    contraction_point = 1
    h5group = ACTIVE_CONTRACTION_GROUP.format(args.alpha, contraction_point) + "/regional"

    write_opt_results_to_h5(args.filename, h5group, args, ini_for_res, for_result_opt, opt_gamma = opt_gamma)




def test_write_parameters():


    from adjoint_contraction_args import setup_adjoint_contraction_parameters, setup_optimization_parameters
    
    
    fname = "test.h5"
    h5file = HDF5File(mpi_comm_world(), "test.h5", "w")

    

    

    def dump_parameters_to_attributes(params, h5group):

        for k,v in params.iteritems():

            if isinstance(v, Parameters) or isinstance(v, dict):
                for k_sub, v_sub in v.iteritems():
                    h5file.attributes(h5group)["{}/{}".format(k,k_sub)] = v_sub

            else:
                h5file.attributes(h5group)[k] = v

    h5group = "test_1"
    par = setup_adjoint_contraction_parameters()
    
    h5file.write(np.array([1.0]), h5group)
    print os.path.abspath(os.path.dirname(par["sim_file"]))
    dump_parameters_to_attributes(par, h5group)
    

    with open("test_parms.yml", 'w') as parfile:
        yaml.dump(par.to_dict(), parfile, default_flow_style=False)

    h5file.close()


if __name__ == "__main__":
    # test_diastolic_output()
    # test_systolic_output()
    # test_multiple_points_output()
    test_write_parameters()
