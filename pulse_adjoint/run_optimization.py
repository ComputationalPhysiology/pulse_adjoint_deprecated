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
from utils import Text, Object, pformat, print_line, print_head, contract_point_exists, get_spaces,  UnableToChangePressureExeption, get_simulated_pressure
from forward_runner import ActiveForwardRunner, PassiveForwardRunner
from optimization_targets import *
from numpy_mpi import *
from adjoint_contraction_args import *
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import minimize_scalar as scipy_minimize_1d
from store_opt_results import write_opt_results_to_h5

try:
    import pyipopt
    has_pyipopt = True
except:
    has_pyipopt = False

try:
    import moola
    has_moola = True
except:
    has_moola = False


def run_passive_optimization(params, patient):
    """
    Main function for the passive phase

    :param dict params: adjoin_contraction_parameters
    :param patient: A patient instance
    :type patient: :py:class`patient_data.Patient`

    **Example of usage**::

      # Setup compiler parameters
      setup_general_parameters()
      params = setup_adjoint_contraction_parameter()
      params['phase'] = 'passive_inflation'
      patient = initialize_patient_data(param['Patient_parameters'], False)
      run_passive_optimization(params, patient)


    """
    

    logger.info(Text.blue("\nRun Passive Optimization"))

    #Load patient data, and set up the simulation
    measurements, solver_parameters, pressure, paramvec = setup_simulation(params, patient)

    rd, paramvec = run_passive_optimization_step(params, 
                                                 patient, 
                                                 solver_parameters, 
                                                 measurements, 
                                                 pressure,
                                                 paramvec)

    logger.info("\nSolve optimization problem.......")
    solve_oc_problem(params, rd, paramvec)


def run_passive_optimization_step(params, patient, solver_parameters, measurements, pressure, paramvec):
    """FIXME! briefly describe function

    :param params: 
    :param patient: 
    :param solver_parameters: 
    :param measurements: 
    :param pressure: 
    :param paramvec: 
    :returns: 
    :rtype: 

    """
    
    # Load targets
    optimization_targets, bcs = load_targets(params, solver_parameters, measurements)
       
    #Initialize the solver for the Forward problem
    for_run = PassiveForwardRunner(solver_parameters, 
                                   pressure, 
                                   bcs,
                                   optimization_targets,
                                   params, 
                                   paramvec)

   
    #Solve the forward problem with guess results (just for printing)
    logger.info(Text.blue("\nForward solution at guess parameters"))
    forward_result, _ = for_run(paramvec, False)
    

    # Update the weights for the functional
    if params["adaptive_weights"]:
        weights = {}
        for k, v in for_run.opt_weights.iteritems():
            weights[k] = v/(10*forward_result["func_value"])
        for_run.opt_weights.update(**weights)
        logger.info("Update weights for functional")
        logger.info(for_run._print_functional())
    
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
    """FIXME! briefly describe function

    :param params: 
    :param patient: 
    :returns: 
    :rtype: 

    """
    
    
    logger.info(Text.blue("\nRun Active Optimization"))

    #Load patient data, and set up the simulation
    measurements, solver_parameters, pressure, gamma = setup_simulation(params, patient)

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
                                                             measurements,
                                                             pressure, 
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
        else:

            # Make sure to do interpolation if that was done earlier
            plv = get_simulated_pressure(params)
            if not plv == measurements["pressure"][i+1]:
                patient.interpolate_data(i+patient.passive_filling_duration-1)
                measurements = get_measurements(params, patient)
                i -= 1
        i += 1

def run_active_optimization_step(params, patient, solver_parameters, measurements, pressure, gamma):
    """FIXME! briefly describe function

    :param params: 
    :param patient: 
    :param solver_parameters: 
    :param measurements: 
    :param pressure: 
    :param gamma: 
    :returns: 
    :rtype: 

    """
    

    #Get initial guess for gamma
    if not params["nonzero_initial_guess"] or params["active_contraction_iteration_number"] == 0:
        # Use zero initial gubess
        zero = Constant(0.0) if gamma.value_size() == 1 \
          else Constant([0.0]*gamma.value_size())

        gamma.assign(zero)
       
    else:
        # Use gamma from the previous point as initial guess
        # Load gamma from previous point
        g_temp = Function(gamma.function_space())
        with HDF5File(mpi_comm_world(), params["sim_file"], "r") as h5file:
            h5file.read(g_temp, "active_contraction/contract_point_{}/optimal_control".format(params["active_contraction_iteration_number"]-1))
        gamma.assign(g_temp)
        

    # Load targets
    optimization_targets, bcs = load_targets(params, solver_parameters, measurements)
    
    for_run = ActiveForwardRunner(solver_parameters,
                                  pressure,
                                  bcs,
                                  optimization_targets,
                                  params,
                                  gamma)

    #Solve the forward problem with guess results (just for printing)
    logger.info(Text.blue("\nForward solution at guess parameters"))
    forward_result, _ = for_run(gamma, False)

    # Update weights so that the initial value of the
    # functional is 0.1
    if params["adaptive_weights"]:
        weights = {}
        for k, v in for_run.opt_weights.iteritems():
            weights[k] = v/(10*forward_result["func_value"])
        for_run.opt_weights.update(**weights)
        logger.info("Update weights for functional")
        logger.info(for_run._print_functional())
    
    # Stop recording
    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True

    
    rd = MyReducedFunctional(for_run, gamma)

    
    # Evaluate the reduced functional in case the solver chrashes at the first point.
    # If this is not done, and the solver crashes in the first point
    # then Dolfin adjoit has no recording and will raise an exception.
    rd(gamma)

    
    return rd, gamma

 
    


def store(params, rd, opt_result):

    from lvsolver import LVSolver
    solver =  LVSolver(rd.for_run.solver_parameters)

    if params["phase"] == PHASES[0]:

        h5group =  PASSIVE_INFLATION_GROUP
        write_opt_results_to_h5(h5group,
                                params,
                                rd.for_res,
                                solver,
                                opt_result)
    else:
        
        h5group =  ACTIVE_CONTRACTION_GROUP.format(params["active_contraction_iteration_number"])
        write_opt_results_to_h5(h5group,
                                params,
                                rd.for_res,
                                solver, 
                                opt_result)


def minimize_1d(f, x0, **kwargs):

    # Initial step size
    dx = np.abs(np.diff(kwargs["bounds"]))[0]/5.0
   
    # Initial functional value
    f_prev = f.func_values_lst[0]

    # If the initial step size is too large, reduce it
    while x0 + dx > kwargs["bounds"][1]:
        dx /= 2
    

    # Evaluate the functional at the new point
    f_cur = f(x0 + dx)
   
    # If the current value is larger than the previous one, try to step in the other direction
    if f_cur > f_prev:
     
        dx *= -1
        while x0 + dx < kwargs["bounds"][0]:
            dx /= 2
        
        f_cur = f(x0 + dx)

    # If this still is true, then the minimum is witin the interval we just checked (assuming convexity).
    if f_cur > f_prev:
       
        
        if x0 - dx > x0:
            a = x0 + dx
            b = x0 - dx
        else:
            a = x0 - dx
            b = x0 + dx
       
        return scipy_minimize_1d(f, bracket = (a,b), **kwargs)

    # Otherwise we step up until the current value if larger then the previous one
    else:
            
        while f_cur < f_prev:

            # If the new value is outside the bounds reduce the step size
            while x0 + dx > kwargs["bounds"][1] or x0 + dx < kwargs["bounds"][0]:
                dx /= 2
               
            
            x0 = x0 + dx
            f_prev_tmp = f_cur
            
            ncrashes = f.nr_crashes
            # Try to evaluate the functional at the new point
            f_cur = f(x0 +dx)

            # Check if the solver chrashed in the evaluation
            if f.nr_crashes > ncrashes:
                # We were not able to evaluate the funcitonal, reduce step size until convergence
                crash = True
                ncrashes = f.nr_crashes
                x0 = x0 - dx
                while crash:
                    
                    dx /= 2
                    x0 = x0 +dx
                    f_cur = f(x0 +dx)
                    
                    if ncrashes == f_cur.nr_crashes:
                        crash = False
                    else:
                        x0 = x0-dx
                    
                    
            # Assign the previous value
            f_prev = f_prev_tmp

        # If f_cur > f_prev we have a interval to search for the minimum (assuming convexity).
        if x0 - dx > x0:
            a = x0
            b = x0 - dx
        else:
            a = x0 - dx
            b = x0
  
        return scipy_minimize_1d(f, bracket = (a,b), **kwargs)
            
            
            
    
    
    
    
        
def solve_oc_problem(params, rd, paramvec):
    """Solve the optimal control problem

    :param params: Application parameters
    :param rd: The reduced functional
    :param paramvec: The control parameter(s)

    """
    

    paramvec_arr = gather_broadcast(paramvec.vector().array())
    opt_params = params["Optimization_parmeteres"]

    if params["phase"] == PHASES[0] and not params["optimize_matparams"]:

        rd(paramvec_arr)
        rd.for_res["initial_control"] = rd.initial_paramvec,
        rd.for_res["optimal_control"] = rd.paramvec
        store(params, rd, {})

    else:

        # Number of control variables
        nvar = len(paramvec_arr)

        if params["phase"] == PHASES[0]:

            lb = np.array([opt_params["matparams_min"]]*nvar)
            ub = np.array([opt_params["matparams_max"]]*nvar)
                
            tol = opt_params["passive_opt_tol"]
            max_iter = opt_params["passive_maxiter"]

        else:

            if params["active_model"] == "active_strain":
                lb = np.array([0.0]*nvar)
                ub = np.array([0.9]*nvar)
            elif params["active_model"] == "active_strain_rossi":
                lb = np.array([-0.9]*nvar)
                ub = np.array([0.0]*nvar)
            else: # Active stress
                lb = np.array([0.0]*nvar)
                ub = np.array([1.0]*nvar)

            tol= opt_params["active_opt_tol"]
            max_iter = opt_params["active_maxiter"]


        
        if nvar == 1:
            # Use 1D optimization method

            kwargs = {"method": opt_params["method_1d"],
                      "bounds":zip(lb,ub)[0],
                      "tol":tol,
                      "options": {"maxiter":max_iter}
            }
            
            t = Timer()
            t.start()
            # Solve the optimization problem
            opt_result = minimize_1d(rd, paramvec_arr[0], **kwargs)

            # scipy_minimize_1d(rd, **kwargs)
            run_time = t.stop()

            opt_result["status"] = ""
            opt_result["message"] = ""
            opt_result["njev"] = rd.nr_der_calls
            opt_result["ncrash"] = rd.nr_crashes
            opt_result["run_time"] = run_time
            opt_result["controls"] = rd.controls_lst
            opt_result["func_vals"] = rd.func_values_lst
            opt_result["forward_times"] = rd.forward_times
            opt_result["backward_times"] = rd.backward_times

            
            print_optimization_report(params, rd.paramvec, rd.initial_paramvec,
                                      rd.ini_for_res, rd.for_res, opt_result)

            for k in ["message", "status", "success"]:
                opt_result.pop(k, None)
        else:
            # Use a gradient based optimization method
            
            if has_pyipopt and opt_params["method"] == "ipopt":

                rd(paramvec_arr)

                lb = opt_params["matparams_min"]
                ub = opt_params["matparams_max"]
                problem = MinimizationProblem(rd, bounds=(lb, ub))
               
                ipopt_parameters = {'maximum_iterations': max_iter, "tol": tol}

                solver = IPOPTSolver(problem, parameters=ipopt_parameters)
                x = solver.solve()

                # Start a timer to measure duration of the optimization
                t = Timer()
                t.start()

                run_time = t.stop()
                
                message_exit_status = {0:"Optimization terminated successfully", 
                                   -1:"Iteration limit exceeded"}
            
                opt_result= {"ncrash":rd.nr_crashes, 
                             "run_time": run_time, 
                             "nfev":rd.iter,
                             "nit":rd.iter,
                             "njev":rd.nr_der_calls,
                             # "status":status,
                             # "message": message_exit_status[status],
                             "x":x,
                             "controls": rd.controls_lst,
                             "func_vals": rd.func_values_lst,
                             "forward_times": rd.forward_times,
                             "backward_times": rd.backward_times}

            elif has_moola and opt_params["method"] == "moola":

                problem = MoolaOptimizationProblem(rd)
                
                paramvec_moola = moola.DolfinPrimalVector(paramvec)
                # solver = moola.NewtonCG(problem, paramvec_moola, options={'gtol': 1e-9,
                #                                                           'maxiter': 20, 
                #                                                           'display': 3, 
                #                                                           'ncg_hesstol': 0})
                
                
                solver = moola.BFGS(problem, paramvec_moola, options={'jtol': 0,
                                                                      'gtol': 1e-9,
                                                                      'Hinit': "default",
                                                                      'maxiter': 100,
                                                                      'mem_lim': 10})
                # solver = moola.NonLinearCG(problem, paramvec_moola, options={'jtol': 0,
                #                                                              'gtol': 1e-9,
                #                                                              'Hinit': "default",
                #                                                              'maxiter': 100,
                #                                                              'mem_lim': 10})
            

                
                t = Timer()
                t.start()
                # Solve the optimization problem
                sol = solver.solve()
                x = sol['control'].data
                
                run_time = t.stop()

                opt_result["x"] = x
                opt_result["status"] = ""
                opt_result["message"] = ""
                opt_result["njev"] = rd.nr_der_calls
                opt_result["ncrash"] = rd.nr_crashes
                opt_result["run_time"] = run_time
                opt_result["controls"] = rd.controls_lst
                opt_result["func_vals"] = rd.func_values_lst
                opt_result["forward_times"] = rd.forward_times
                opt_result["backward_times"] = rd.backward_times
                
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
                
            
                kwargs = {"method": method,
                          "constraints": cons, 
                          # "bounds":zip(lb,ub),
                          "jac": rd.derivative,
                          "tol":tol,
                          "options": {"disp": opt_params["disp"],
                                      # "iprint": 2,
                                      "ftol": 1e-16,
                                      "maxiter":max_iter}
                }
                # if method == "SLSQP":
                #     kwargs["constraints"] = cons
                # else:
                #     kwargs["bounds"] = zip(lb,ub)
                

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
                opt_result["forward_times"] = rd.forward_times
                opt_result["backward_times"] = rd.backward_times

                print_optimization_report(params, rd.paramvec, rd.initial_paramvec,
                                          rd.ini_for_res, rd.for_res, opt_result)

                for k in ["message", "status", "success"]:
                    opt_result.pop(k, None)

        x = np.array([opt_result.pop("x")]) if nvar == 1 else gather_broadcast(opt_result.pop("x"))
        assign_to_vector(paramvec.vector(), gather_broadcast(x))
        logger.info(Text.blue("\nForward solution at optimal parameters"))
        val = rd.for_run(paramvec, False)
          
        rd.for_res["initial_control"] = rd.initial_paramvec,
        rd.for_res["optimal_control"] = rd.paramvec
        
        
        
        
        store(params, rd, opt_result)

def print_optimization_report(params, opt_controls, init_controls, 
                              ini_for_res, opt_for_res, opt_result = None):

    from numpy_mpi import gather_broadcast

    if opt_result:
        logger.info("\nOptimization terminated...")
        logger.info("\tExit status {}".format(opt_result["status"]))
        logger.info("\tMessage: {}".format(opt_result["message"]))
        logger.info("\tFunction Evaluations: {}".format(opt_result["nfev"]))
        logger.info("\tGradient Evaluations: {}".format(opt_result["njev"]))
        logger.info("\tNumber of iterations: {}".format(opt_result["nit"]))
        logger.info("\tNumber of crashes: {}".format(opt_result["ncrash"]))
        logger.info("\tRun time: {:.2f} seconds".format(opt_result["run_time"]))

    logger.info("\nFunctional Values")
    logger.info(" "*7+"\t"+print_head(ini_for_res, False))
    logger.info("{:7}\t{}".format("Initial", print_line(ini_for_res)))
    logger.info("{:7}\t{}".format("Optimal", print_line(opt_for_res)))

    if params["phase"] == PHASES[0]:
        logger.info("\nMaterial Parameters")
        logger.info("Initial {}".format(init_controls))
        logger.info("Optimal {}".format(gather_broadcast(opt_controls.vector().array())))
    else:
        logger.info("\nContraction Parameter")
        logger.info("\tMin\tMean\tMax")
        logger.info("Initial\t{:.5f}\t{:.5f}\t{:.5f}".format(init_controls.min(), 
                                                             init_controls.mean(), 
                                                             init_controls.max()))
        opt_controls_arr = gather_broadcast(opt_controls.vector().array())
        logger.info("Optimal\t{:.5f}\t{:.5f}\t{:.5f}".format(opt_controls_arr.min(), 
                                                             opt_controls_arr.mean(), 
                                                             opt_controls_arr.max()))


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
    biv = params["Patient_parameters"]["mesh_type"] == "biv"

    # Load boundary conditions
    bcs = {}

    pressure = measurements["pressure"]
    if biv:
        rv_pressure = measurements["rv_pressure"]
        
    
    if params["phase"] == PHASES[1]:
        pressure = pressure[acin: 2 + acin]
        if biv:
            rv_pressure = rv_pressure[acin: 2 + acin]
        
    bcs["pressure"] = pressure
    if biv:
        bcs["rv_pressure"] = rv_pressure
    


    # Load the target data into dofin functions
    for key, val in params["Optimization_targets"].iteritems():

        # If target is included in the optimization
        if val:
            # Load the target data
            for it,p in enumerate(pressure):
                optimization_targets[key].load_target_data(measurements[key], it+acin)
                
    return optimization_targets, bcs


def get_optimization_targets(params, solver_parameters):
    """FIXME! briefly describe function

    :param params: 
    :param solver_parameters: 
    :returns: 
    :rtype: 

    """
    

    p = params["Optimization_targets"]
    mesh = solver_parameters["mesh"]
    if params["phase"] == PHASES[0]:
        reg_par = params["Passive_optimization_weigths"]["regularization"]
    else:
        reg_par = params["Active_optimization_weigths"]["regularization"]

    targets = {"regularization": Regularization(mesh,
                                                params["gamma_space"],
                                                reg_par)}

    if p["volume"]:
        
        if params["Patient_parameters"]["mesh_type"] == "biv":
            marker = "ENDO_LV"
        else:
            marker = "ENDO"
            
        dS = Measure("exterior_facet",
                     subdomain_data = solver_parameters["facet_function"],
                     domain = mesh)(solver_parameters["markers"][marker][0])
        
        targets["volume"] = VolumeTarget(mesh,dS, "LV")

    if p["rv_volume"]:
            
        dS = Measure("exterior_facet",
                     subdomain_data = solver_parameters["facet_function"],
                     domain = mesh)(solver_parameters["markers"]["ENDO_RV"][0])
        
        targets["rv_volume"] = VolumeTarget(mesh,dS, "RV")

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
        
    
def load_targets(params, solver_parameters, measurements):
    """FIXME! briefly describe function

    :param dict params: 
    :param dict solver_parameters: 
    :param dict measurements: 
    :returns: A tuple containing 1. a dictionary with
              optimization targets and 2. boundary conditions
    :rtype: tuple

    """
    
    
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

    return optimization_targets, bcs
