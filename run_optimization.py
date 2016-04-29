from dolfin import *
from dolfin_adjoint import *
from setup_optimization import setup_simulation, logger, MyReducedFunctional
from utils import Text, Object, pformat, print_optimization_report, contract_point_exists, get_spaces
from forward_runner import ActiveForwardRunner, PassiveForwardRunner

from numpy_mpi import *
from adjoint_contraction_args import *
from scipy.optimize import minimize as scipy_minimize
from store_opt_results import write_opt_results_to_h5

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
    

    

    mesh = solver_parameters["mesh"]
    spaces = get_spaces(mesh)
    crl_basis = (patient.e_circ, patient.e_rad, patient.e_long)
     
    
    #Solve calls are not registred by libajoint
    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True
    
    # Load target data
    target_data = load_target_data(measurements, params, spaces)

    
    # Start recording for dolfin adjoint 
    logger.debug(Text.yellow("Start annotating"))
    parameters["adjoint"]["stop_annotating"] = False

       
    #Initialize the solver for the Forward problem
    for_run = PassiveForwardRunner(solver_parameters, 
                                   p_lv, 
                                   target_data,  
                                   patient.ENDO,
                                   crl_basis,
                                   params, 
                                   spaces, 
                                   paramvec)

    #Solve the forward problem with guess results (just for printing)
    logger.info(Text.blue("\nForward solution at guess parameters"))
    forward_result, _ = for_run(paramvec, True)
    

    # Stop recording
    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True

    # Initialize MyReducedFuctional
    rd = MyReducedFunctional(for_run, paramvec)
    
    return rd, paramvec


def run_active_optimization(params, patient):
    
    logger.info(Text.blue("\nRun Active Optimization"))

    #Load patient data, and set up the simulation
    measurements, solver_parameters, p_lv, gamma = setup_simulation(params, patient)
    
    # Loop over contract points
    for i in range(patient.num_contract_points):
        params["active_contraction_iteration_number"] = i
        if not contract_point_exists(params):
            
            rd, gamma = run_active_optimization_step(params, patient, 
                                                     solver_parameters, 
                                                     measurements, p_lv, 
                                                     gamma)


            logger.info("\nSolve optimization problem.......")
            solve_oc_problem(params, rd, gamma)
            adj_reset()

def run_active_optimization_step(params, patient, solver_parameters, measurements, p_lv, gamma):

    
    # Circumferential, radial and logitudinal basis vectors
    crl_basis = (patient.e_circ, patient.e_rad, patient.e_long)
    mesh = solver_parameters["mesh"]

    # Initialize spaces
    spaces = get_spaces(mesh)
    

    #Get initial guess for gamma
    if not params["nonzero_initial_guess"] or params["active_contraction_iteration_number"] == 0:
        # Use zero initial guess
        gamma.assign(Constant(0.0))
    else:
        # Use gamma from the previous point as initial guess
        # Load gamma from previous point
        with HDF5File(mpi_comm_world(), params["sim_file"], "r") as h5file:
            h5file.read(gamma, "alpha_{}/active_contraction/contract_point_{}/parameters/activation_parameter_function/".format(params["alpha"], params["active_contraction_iteration_number"]-1))
  
        

    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True
    
    target_data = load_target_data(measurements, params, spaces)

    logger.debug(Text.yellow("Start annotating"))
    parameters["adjoint"]["stop_annotating"] = False
   
    for_run = ActiveForwardRunner(solver_parameters,
                                  p_lv,
                                  target_data,
                                  params,
                                  gamma, 
                                  patient, 
                                  spaces)

    #Solve the forward problem with guess results (just for printing)
    logger.info(Text.blue("\nForward solution at guess parameters"))
    forward_result, _ = for_run(gamma, False)

    # Stop recording
    logger.debug(Text.yellow("Stop annotating"))
    parameters["adjoint"]["stop_annotating"] = True

    # Compute the functional as a pure function of gamma
    rd = MyReducedFunctional(for_run, gamma)
            
    return rd, gamma

 
    


def store(params, rd, opt_controls, opt_result=None):

    if params["phase"] == PHASES[0]:

        h5group =  PASSIVE_INFLATION_GROUP.format(params["alpha_matparams"])
        write_opt_results_to_h5(h5group, params, rd.ini_for_res, rd.for_res, 
                                opt_matparams = opt_controls, 
                                opt_result = opt_result)
    else:
        
        h5group =  ACTIVE_CONTRACTION_GROUP.format(params["alpha"], params["active_contraction_iteration_number"])
        write_opt_results_to_h5(h5group, params, rd.ini_for_res, rd.for_res, 
                                opt_gamma = opt_controls, opt_result = opt_result)


def solve_oc_problem(params, rd, paramvec):

    paramvec_arr = gather_broadcast(paramvec.vector().array())

    if params["phase"] == PHASES[0] and not params["optimize_matparams"]:
        rd(paramvec_arr)
        store(params, rd, paramvec)

    else:
        

        opt_params = params["Optimization_parmeteres"]
        kwargs = {"method": opt_params["method"],
                  "jac": rd.derivative,
                  "options": {"disp": opt_params["disp"]}
                  }

        if params["phase"] == PHASES[0]:

            kwargs["tol"] = opt_params["passive_opt_tol"]
            kwargs["bounds"] = [(opt_params["matparams_min"], 
                                 opt_params["matparams_max"])]*len(paramvec_arr)
            kwargs["options"]["maxiter"] = opt_params["passive_maxiter"]
        else:

            kwargs["tol"] = opt_params["active_opt_tol"]
            kwargs["bounds"] = [(0, opt_params["gamma_max"])]*len(paramvec_arr)
            kwargs["options"]["maxiter"] = opt_params["active_maxiter"]

        # Start a timer to measure duration of the optimization
        t = Timer()
        t.start()
        # Solve the optimization problem
        opt_result = scipy_minimize(rd,paramvec_arr, **kwargs)
        run_time = t.stop()
        opt_result["ncrash"] = rd.nr_crashes
        opt_result["run_time"] = run_time
        assign_to_vector(paramvec.vector(), opt_result.x)

        
        print_optimization_report(params, rd.paramvec, rd.initial_paramvec,
                                  rd.ini_for_res, rd.for_res, opt_result)
        logger.info(Text.blue("\nForward solution at optimal parameters"))
        val = rd.for_run(paramvec, False)
        store(params, rd, paramvec, opt_result)



def load_target_data(measurements, params, spaces):
    logger.debug(Text.blue("Loading Target Data"))
        
    def get_strain(newfunc, i, it):
        assign_to_vector(newfunc.vector(), np.array(measurements.strain[i][it]))


    def get_volume(newvol, it):
        assign_to_vector(newvol.vector(), np.array([measurements.volume[it]]))

    # The target data is put into functions so that Dolfin-adjoint can properly record it.
 
    # Store target strains and volumes
    target_strains = []
    target_vols = []

    acin = params["active_contraction_iteration_number"]

    if params["phase"] == PHASES[0]:
        pressures = measurements.pressure
    else:
        pressures = measurements.pressure[acin: 2 + acin]
       

    logger.info(Text.blue("Load target data"))
    logger.info("\tLV Pressure (cPa) \tLV Volume (mL)")


    for it, p in enumerate(pressures):
        
        if params["use_deintegrated_strains"]:
            newfunc = Function(spaces.strainfieldspace, name = \
                               "strain_{}".format(args.active_contraction_iteration_number+it))
          
            assign_to_vector(newfunc.vector(), \
                             gather_broadcast(measurements.strain_deintegrated[acin+it].array()))
            
            target_strains.append(newfunc)
            
        else:
            strains_at_pressure = []
            for i in STRAIN_REGION_NUMS:
                newfunc = Function(spaces.strainspace, name = "strain_{}_{}".format(acin+it, i))
                get_strain(newfunc, i, acin+it)
                strains_at_pressure.append(newfunc)

            target_strains.append(strains_at_pressure)

        newvol = Function(spaces.r_space, name = "newvol")
        get_volume(newvol, acin+it)
        target_vols.append(newvol)

        logger.info("\t{:.3f} \t\t\t{:.3f}".format(p,gather_broadcast(target_vols[-1].vector().array())[0]))



    target_data = Object()
    target_data.target_strains = target_strains
    target_data.target_vols = target_vols
    target_data.target_pressure = pressures


    return target_data
