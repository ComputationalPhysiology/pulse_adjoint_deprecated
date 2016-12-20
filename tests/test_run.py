"""
Regression tests
This is just some basic tests to which runs through 
each function in the pipeline which can be used to check
that new changes does not break the code.
"""
import pulse_adjoint as pa
from patient_data import TestPatient

pa.setup_optimization.setup_general_parameters()
params = pa.setup_optimization.setup_adjoint_contraction_parameters()

patient = TestPatient()

def test_setup_passive():

    params["phase"] = pa.adjoint_contraction_args.PHASES[0]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)

def test_passive_optimzation_targets():

    params["phase"] = pa.adjoint_contraction_args.PHASES[0]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)
    
    # Load optimization targets
    optimization_targets \
        = pa.run_optimization.get_optimization_targets(params,
                                                       solver_parameters)

    # Load target data
    optimization_targets, bcs \
        = pa.run_optimization.load_target_data(measurements, params,
                                               optimization_targets)

def test_passive_forward_runner():

    params["phase"] = pa.adjoint_contraction_args.PHASES[0]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)
    
    # Load optimization targets
    optimization_targets \
        = pa.run_optimization.get_optimization_targets(params,
                                                       solver_parameters)

    # Load target data
    optimization_targets, bcs \
        = pa.run_optimization.load_target_data(measurements, params,
                                               optimization_targets)

    #Initialize the solver for the Forward problem
    for_run = pa.forward_runner.PassiveForwardRunner(solver_parameters, 
                                                     pressure, 
                                                     bcs,
                                                     optimization_targets,
                                                     params, 
                                                     paramvec)
    

def test_passive_optimization():

    params["phase"] = pa.adjoint_contraction_args.PHASES[0]
    params["matparams_space"] = "CG_1"
    params["Optimization_parmeteres"]["passive_maxiter"] = 1
    pa.run_optimization.run_passive_optimization(params, patient)
    
    

def test_setup_active():

    if not pa.utils.passive_inflation_exists(params):
        params["phase"] =  pa.adjoint_contraction_args.PHASES[0]
        params["optimize_matparams"] = False
        pa.run_optimization.run_passive_optimization(params, patient)
        
    params["phase"] = pa.adjoint_contraction_args.PHASES[1]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)

def test_active_optimzation_targets():

    if not pa.utils.passive_inflation_exists(params):
        params["phase"] =  pa.adjoint_contraction_args.PHASES[0]
        params["optimize_matparams"] = False
        pa.run_optimization.run_passive_optimization(params, patient)
        
    params["phase"] = pa.adjoint_contraction_args.PHASES[1]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)
    
    # Load optimization targets
    optimization_targets \
        = pa.run_optimization.get_optimization_targets(params,
                                                       solver_parameters)

    # Load target data
    optimization_targets, bcs \
        = pa.run_optimization.load_target_data(measurements, params,
                                               optimization_targets)
def test_active_forward_runner():

    if not pa.utils.passive_inflation_exists(params):
        params["phase"] =  pa.adjoint_contraction_args.PHASES[0]
        params["optimize_matparams"] = False
        pa.run_optimization.run_passive_optimization(params, patient)
        
    params["phase"] = pa.adjoint_contraction_args.PHASES[1]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)
    
    # Load optimization targets
    optimization_targets \
        = pa.run_optimization.get_optimization_targets(params,
                                                       solver_parameters)

    # Load target data
    optimization_targets, bcs \
        = pa.run_optimization.load_target_data(measurements, params,
                                               optimization_targets)

    #Initialize the solver for the Forward problem
    for_run = pa.forward_runner.ActiveForwardRunner(solver_parameters, 
                                                    pressure, 
                                                    bcs,
                                                    optimization_targets,
                                                    params, 
                                                    paramvec)


def test_active_optimization():

    if not pa.utils.passive_inflation_exists(params):
        params["phase"] =  pa.adjoint_contraction_args.PHASES[0]
        params["optimize_matparams"] = False
        pa.run_optimization.run_passive_optimization(params, patient)
        
    params["phase"] = pa.adjoint_contraction_args.PHASES[1]
    params["gamma_space"] = "CG_1"
    params["Optimization_parmeteres"]["active_maxiter"] = 1
    pa.run_optimization.run_active_optimization(params, patient)

