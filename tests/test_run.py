import pulse_adjoint as pa
from patient_data import TestPatient

pa.setup_optimization.setup_general_parameters()
params = pa.setup_optimization.setup_adjoint_contraction_parameters()

patient = TestPatient()

def test_setup_passive():

    params["phase"] = pa.adjoint_contraction_args.PHASES[0]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)

def test_optimzation_targets():

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

    rd = pa.setup_optimization.MyReducedFunctional(for_run, paramvec)
    rd(paramvec)
    
    
    

def a_test_setup_active():
    params["phase"] = pa.adjoint_contraction_args.PHASES[1]
    measurements, solver_parameters, pressure, paramvec \
        = pa.setup_optimization.setup_simulation(params, patient)

def test_store():
    pass


def test_active_forward_runner():
    pass

