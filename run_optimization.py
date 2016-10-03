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
from setup_optimization import setup_simulation, logger, MyReducedFunctional, get_measurements
from utils import Text, Object, pformat, print_optimization_report, contract_point_exists, get_spaces,  UnableToChangePressureExeption
from forward_runner import ActiveForwardRunner, PassiveForwardRunner
from optimization_targets import *
from numpy_mpi import *
from adjoint_contraction_args import *
from scipy.optimize import minimize as scipy_minimize
from store_opt_results import write_opt_results_to_h5

try:
    import pyipopt
    has_pyipopt = True
except:
    has_pyipopt = False


def run_passive_optimization(params, patient):

    logger.info(Text.blue("\nRun Passive Optimization"))

    #Load patient data, and set up the simulation
    measurements, solver_parameters, p_lv, paramvec = setup_simulation(params, patient)

    rd, paramvec = run_passive_optimization_step(params, 
                                                 patient, 
                                                 solver_parameters, 
                                                 measurements, 
                                                 p_lv, paramvec)

    logger.info("\nSolve optimization problem.......")
    solve_oc_problem(params, rd, paramvec)


def run_passive_optimization_step(params, patient, solver_parameters, measurements, p_lv, paramvec):
    
    #Solve calls are not registred by libajoint
    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True

    
    # Load optimization targets
    optimization_targets = get_optimization_targets(params, solver_parameters)

    # Load target data
    optimization_targets, bcs = \
        load_target_data(measurements, params, optimization_targets)

    
    
    
    # Start recording for dolfin adjoint 
    logger.debug(Text.yellow("Start annotating"))
    parameters["adjoint"]["stop_annotating"] = False

       
    #Initialize the solver for the Forward problem
    for_run = PassiveForwardRunner(solver_parameters, 
                                   p_lv, 
                                   bcs,
                                   optimization_targets,
                                   params, 
                                   paramvec)

    #Solve the forward problem with guess results (just for printing)
    logger.info(Text.blue("\nForward solution at guess parameters"))
    forward_result, _ = for_run(paramvec, False)
    

    # Stop recording
    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True

    # Initialize MyReducedFuctional
    rd = MyReducedFunctional(for_run, paramvec)

    
    # Evaluate the reduced functional in case the solver chrashes at the first point.
    # If this is not done, and the solver crashes in the first point
    # then Dolfin adjoit has no recording and will raise an exception.
    rd(paramvec)
    
    return rd, paramvec


def run_active_optimization(params, patient):
    
    logger.info(Text.blue("\nRun Active Optimization"))

    #Load patient data, and set up the simulation
    measurements, solver_parameters, p_lv, gamma = setup_simulation(params, patient)

    # Loop over contract points
    i = 0
    # for i in range(patient.num_contract_points):
    while i < patient.num_contract_points:
        params["active_contraction_iteration_number"] = i
        if not contract_point_exists(params):

            # Number of times we have interpolated in order
            # to be able to change the pressure
            attempts = 0
            pressure_change = False
            
            while (not pressure_change and attempts < 8):
               
                try:
                    rd, gamma = run_active_optimization_step(params, patient, 
                                                             solver_parameters, 
                                                             measurements, p_lv, 
                                                             gamma)
                except UnableToChangePressureExeption:
                    logger.info("Unable to change pressure. Exception caught")

                    logger.info("Lets interpolate. Add one extra point")
                    patient.interpolate_data(i+patient.passive_filling_duration-1)

                    # Update the measurements
                    measurements = get_measurements(params, patient)
                    
                    attempts += 1
                    
                else:
                    pressure_change = True

                    logger.info("\nSolve optimization problem.......")
                    solve_oc_problem(params, rd, gamma)
                    adj_reset()
         
            if not pressure_change:
                raise RuntimeError("Unable to increasure")
            
        i += 1

def run_active_optimization_step(params, patient, solver_parameters, measurements, p_lv, gamma):

    #Get initial guess for gamma
    if not params["nonzero_initial_guess"] or params["active_contraction_iteration_number"] == 0:
        # Use zero initial guess
        zero = Constant(0.0) if gamma.value_size() == 1 \
          else Constant([0.0]*gamma.value_size())

        gamma.assign(zero)
       
    else:
        # Use gamma from the previous point as initial guess
        # Load gamma from previous point
        with HDF5File(mpi_comm_world(), params["sim_file"], "r") as h5file:
            h5file.read(gamma, "alpha_{}/active_contraction/contract_point_{}/parameters/activation_parameter_function/".format(params["alpha"], params["active_contraction_iteration_number"]-1))
  
        

    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True
    
    # Load optimization targets
    optimization_targets = get_optimization_targets(params, solver_parameters)

    # Load target data
    optimization_targets, bcs = \
        load_target_data(measurements, params, optimization_targets)

    logger.debug(Text.yellow("Start annotating"))
    parameters["adjoint"]["stop_annotating"] = False
   
    for_run = ActiveForwardRunner(solver_parameters,
                                  p_lv,
                                  bcs,
                                  optimization_targets,
                                  params,
                                  gamma)

    #Solve the forward problem with guess results (just for printing)
    logger.info(Text.blue("\nForward solution at guess parameters"))
    forward_result, _ = for_run(gamma, False)

    # Stop recording
    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True

    # Compute the functional as a pure function of gamma
    rd = MyReducedFunctional(for_run, gamma)

    
    # Evaluate the reduced functional in case the solver chrashes at the first point.
    # If this is not done, and the solver crashes in the first point
    # then Dolfin adjoit has no recording and will raise an exception.
    rd(gamma)

    
    return rd, gamma

 
    


def store(params, rd, opt_controls, opt_result=None):

    if params["phase"] == PHASES[0]:

        h5group =  PASSIVE_INFLATION_GROUP
        write_opt_results_to_h5(h5group, params, rd.ini_for_res, rd.for_res, 
                                opt_matparams = opt_controls, 
                                opt_result = opt_result)
    else:
        
        h5group =  ACTIVE_CONTRACTION_GROUP.format(params["active_contraction_iteration_number"])
        write_opt_results_to_h5(h5group, params, rd.ini_for_res, rd.for_res, 
                                opt_gamma = opt_controls, opt_result = opt_result)


def solve_oc_problem(params, rd, paramvec):

    paramvec_arr = gather_broadcast(paramvec.vector().array())
    opt_params = params["Optimization_parmeteres"]

    if params["phase"] == PHASES[0] and not params["optimize_matparams"]:
        rd(paramvec_arr)
        store(params, rd, paramvec)

    else:

        # Number of control variables
        nvar = len(paramvec_arr)

        if params["phase"] == PHASES[0]:

            lb = np.array([opt_params["matparams_min"]]*nvar)
            ub = np.array([opt_params["matparams_max"]]*nvar)

            if opt_params["fix_a"]:
                lb[0] = ub[0] = paramvec_arr[0]
            if opt_params["fix_a_f"]:
                lb[1] = ub[1] = paramvec_arr[1]
            if opt_params["fix_b"]:
                lb[2] = ub[2] = paramvec_arr[2]
            if opt_params["fix_b_f"]:
                lb[3] = ub[3] = paramvec_arr[3]

                
            tol = opt_params["passive_opt_tol"]
            max_iter = opt_params["passive_maxiter"]

        else:

            if params["active_model"] == "active_strain":
                lb = np.array([0.0]*nvar)
                ub = np.array([0.3]*nvar)
            elif params["active_model"] == "active_strain_rossi":
                lb = np.array([-0.3]*nvar)
                ub = np.array([0.0]*nvar)
            else: # Active stress
                lb = np.array([0.0]*nvar)
                ub = np.array([1.0]*nvar)

            tol= opt_params["active_opt_tol"]
            max_iter = opt_params["active_maxiter"]

        
        if has_pyipopt and opt_params["method"] == "ipopt":

            # Bounds
            lb = np.array([opt_params["matparams_min"]]*nvar)
            ub = np.array([opt_params["matparams_max"]]*nvar)
 
            # No constraits 
            nconstraints = 0
            constraints_nnz = nconstraints * nvar
            empty = np.array([], dtype=float)
            clb = empty
            cub = empty

            # The constraint function, should do nothing
            def fun_g(x, user_data=None):
                return empty

            # The constraint Jacobian
            def jac_g(x, flag, user_data=None):
                if flag:
                    rows = np.array([], dtype=int)
                    cols = np.array([], dtype=int)
                    return (rows, cols)
                else:
                    return empty

            J  = rd.__call__
            dJ = rd.derivative

            
            nlp = pyipopt.create(nvar,              # number of control variables
                                 lb,                # lower bounds on control vector
                                 ub,                # upper bounds on control vector
                                 nconstraints,      # number of constraints
                                 clb,               # lower bounds on constraints,
                                 cub,               # upper bounds on constraints,
                                 constraints_nnz,   # number of nonzeros in the constraint Jacobian
                                 0,                 # number of nonzeros in the Hessian
                                 J,                 # to evaluate the functional
                                 dJ,                # to evaluate the gradient
                                 fun_g,             # to evaluate the constraints
                                 jac_g)             # to evaluate the constraint Jacobian

                                 
            
            nlp.num_option('tol', tol)
            nlp.int_option('max_iter', max_iter)
            pyipopt.set_loglevel(1)                 # turn off annoying pyipopt logging

            nlp.str_option("print_timing_statistics", "yes")
            nlp.str_option("warm_start_init_point", "yes")

            print_level = 6 if logger.level < INFO else 4

            if mpi_comm_world().rank > 0:
                nlp.int_option('print_level', 0)    # disable redundant IPOPT output in parallel
            else:
                nlp.int_option('print_level', print_level)    # very useful IPOPT output

            # Do an initial solve to put something in the cache
            rd(paramvec_arr)

            # Start a timer to measure duration of the optimization
            t = Timer()
            t.start()

            # Solve optimization problem with initial guess
            x, zl, zu, constraint_multipliers, obj, status = nlp.solve(paramvec_arr)
            
            run_time = t.stop()

            message_exit_status = {0:"Optimization terminated successfully", 
                                   -1:"Iteration limit exceeded"}
            
            opt_result= {"ncrash":rd.nr_crashes, 
                         "run_time": run_time, 
                         "nfev":rd.iter,
                         "nit":rd.iter,
                         "njev":rd.nr_der_calls,
                         "status":status,
                         "message": message_exit_status[status],
                         "obj":obj}
            
            nlp.close()
            

        else:
            
            if opt_params["method"] == "ipopt":
                logger.Warning("Warning: Ipopt is not installed. Use SLSQP")
                method = "SLSQP"
            else:
                method = opt_params["method"]
                
            def lowerbound_constraint(m):
                return m - lb

            def upperbound_constraint(m):
                return ub - m



            cons = ({"type": "ineq", "fun": lowerbound_constraint},
                    {"type": "ineq", "fun": upperbound_constraint})

            if params["phase"] == PHASES[0] and \
               params["linear_matparams_ratio"] > 0:

                # We put a constaint on the ration between a and a_f
                def ratio_constraint(m):
                    return m[0]/m[1] - float(params["linear_matparams_ratio"])
                cons = (cons + ({"type": "eq",
                                "fun": ratio_constraint}, {}))[:-1]

                logger.info("Force ratio a/a_f = {}".format(params["linear_matparams_ratio"]))
                
                
            
            kwargs = {"method": method,
                      "constraints": cons, 
                      # "bounds":zip(lb,ub),
                      "jac": rd.derivative,
                      "tol":tol,
                      "options": {"disp": opt_params["disp"],
                                  "maxiter":max_iter}
                      }


            # Start a timer to measure duration of the optimization
            t = Timer()
            t.start()
            # Solve the optimization problem
            opt_result = scipy_minimize(rd,paramvec_arr, **kwargs)
            run_time = t.stop()
            opt_result["ncrash"] = rd.nr_crashes
            opt_result["run_time"] = run_time
            opt_result["controls"] = rd.controls_lst
            opt_result["func_vals"] = rd.func_values_lst
            x = opt_result.x

        assign_to_vector(paramvec.vector(), x)

        
        print_optimization_report(params, rd.paramvec, rd.initial_paramvec,
                                  rd.ini_for_res, rd.for_res, opt_result)
        logger.info(Text.blue("\nForward solution at optimal parameters"))
        val = rd.for_run(paramvec, False)
        store(params, rd, paramvec, opt_result)



def load_target_data(measurements, params, optimization_targets):
    """Load the target data into dolfin functions.
    The target data will be loaded into the optiization_targets

    :param measurements: The target measurements
    :param params: Application parameters
    :param optimization_targer: A dictionary with the targets
    :returns: object with target data
    :rtype: object
    """

    logger.debug(Text.blue("Loading Target Data"))

    # The point in the acitve phase (0 if passive)
    acin = params["active_contraction_iteration_number"]

    # Load boundary conditions
    bcs = {}
    if params["phase"] == PHASES[0]:
        pressure = measurements["pressure"]
        # seg_verts = measurements.seg_verts
    else:
        pressure = measurements["pressure"][acin: 2 + acin]
        # seg_verts = measurements.seg_verts[acin: 2 + acin]
        
    bcs["pressure"] = pressure
    # bcs["seg_verts"] = seg_verts

    # Load the target data into dofin functions
    for key, val in params["Optimization_targets"].iteritems():

        # If target is included in the optimization
        if val:
            # Load the target data
            for it,p in enumerate(pressure):
                optimization_targets[key].load_target_data(measurements[key], it+acin)

    return optimization_targets, bcs


def get_optimization_targets(params, solver_parameters):

    p = params["Optimization_targets"]
    mesh = solver_parameters["mesh"]

    targets = {"regularization": Regularization(mesh,
                                                params["gamma_space"],
                                                params["reg_par"])}

    if p["volume"]:
        
        dS = Measure("exterior_facet",
                     subdomain_data = solver_parameters["facet_function"],
                     domain = mesh)(solver_parameters["markers"]["ENDO"][0])
        
        targets["volume"] = VolumeTarget(mesh,dS)

    if p["regional_strain"]:

        dX = Measure("dx",
                     subdomain_data = solver_parameters["mesh_function"],
                     domain = mesh)
        
        targets["regional_strain"] = \
            RegionalStrainTarget(mesh,
                                 solver_parameters["crl_basis"],
                                 dX,
                                 solver_parameters["strain_weights"])
    
        

    return targets
        
    
