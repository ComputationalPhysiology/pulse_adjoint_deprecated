"""
Test that saving optimization results to file
works as is should.
"""
import dolfin, dolfin_adjoint
from mesh_generation.mesh_utils import load_geometry_from_h5

from pulse_adjoint.setup_optimization import (make_solver_params,
                                              get_measurements,
                                              setup_adjoint_contraction_parameters,
                                              setup_general_parameters)

from pulse_adjoint.forward_runner import PassiveForwardRunner
from pulse_adjoint.run_optimization import load_targets
from pulse_adjoint.pa_io import write_opt_results_to_h5
from pulse_adjoint import LVTestPatient


patient = LVTestPatient()


def test_store():

    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    measurements = get_measurements(params, patient)
    
    solver_parameters, pressure, control = make_solver_params(params, patient)
    
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
    for_run.update_targets(0, dolfin.split(w)[0], control)
    for_run.states = [w.copy(True)]
    for_res = for_run._make_forward_result([0.0], [functional*dolfin_adjoint.dt[0.0]])

    for_res["initial_control"] = dolfin.Vector(control.vector())
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
    

if __name__ == "__main__":
    test_store()
