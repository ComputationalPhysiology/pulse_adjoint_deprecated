
from campass.run_optimization import run_passive_optimization_step, run_active_optimization_step, run_passive_optimization
from campass.setup_optimization import initialize_patient_data, setup_simulation
from campass.adjoint_contraction_args import *
from campass.utils import Text, pformat, passive_inflation_exists
from test_utils import setup_params, passive_taylor_test, active_taylor_test, passive_test_functional
from dolfin_adjoint import replay_dolfin, adj_reset, adj_html

alphas = [0.0, 0.5, 1.0]



def test_passive():

    params = setup_params()

    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
    
    logger.info(Text.blue("\nTest Passive Optimization"))

    logger.info(pformat(params.to_dict()))


    params["phase"] = "passive_inflation"
    measurements, solver_parameters, p_lv, paramvec = \
      setup_simulation(params, patient)

    
    control, rd, for_run, forward_result = \
      run_passive_optimization_step(params, 
                                    patient, 
                                    solver_parameters, 
                                    measurements, 
                                    p_lv, paramvec)
    # Dump html visualization of the forward and adjoint system
    adj_html("passive_forward.html", "forward")
    adj_html("passive_adjoint.html", "adjoint")

    # Replay the forward run, i.e make sure that the recording is correct.
    assert replay_dolfin(tol=1e-12)
    
    # Test that the gradient is correct
    passive_taylor_test(rd, control)
    
    # Test that the reduced functional gives the same value as the forward run.
    passive_test_functional(rd, paramvec, for_run, params)
    


def test_active():

    params = setup_params()

    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
    
    logger.info(Text.blue("\nTest Passive Optimization"))

    logger.info(pformat(params.to_dict()))

    if not passive_inflation_exists(params):
        params["phase"] = "passive_inflation"
        params["optimize_matparams"] = False
        run_passive_optimization(params, patient)
        adj_reset()

    params["phase"] = "active_contraction"
    measurements, solver_parameters, p_lv, gamma = \
      setup_simulation(params, patient)
    
    rd, gamma = run_active_optimization_step(params, 
                                             patient, 
                                             solver_parameters, 
                                             measurements, p_lv, 
                                             gamma)

    # Dump html visualization of the forward and adjoint system
    adj_html("active_forward.html", "forward")
    adj_html("active_adjoint.html", "adjoint")

    # Replay the forward run, i.e make sure that the recording is correct.
    replay_dolfin(tol=1e-12)
    
    # Test that the gradient is correct
    active_taylor_test(rd, gamma)

    # There is no need to test the functional since the 
    # reduced functional uses the forward run for evaluation

    

if __name__ == "__main__":
    test_passive()
    test_active()
