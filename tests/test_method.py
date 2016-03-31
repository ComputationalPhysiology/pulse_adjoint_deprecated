from dolfin import *
from dolfin_adjoint import *
from campass.run_optimization import run_passive_optimization_step, run_active_optimization_step
from campass.setup_optimization import setup_adjoint_contraction_parameters, initialize_patient_data, setup_simulation, setup_general_parameters
from campass.adjoint_contraction_args import *
from campass.utils import Object, Text, pformat

alphas = [0.0, 0.5, 1.0]


def active_taylor_test(Jhat, m0, Jm0, DJm0):
    #paramvec = controls.data()
    d = np.array([1.0]*len(m0)) #perturbation direction
    grad_errors = []
    no_grad_errors = []
   
    epsilons = [0.05, 0.025, 0.0125]

    #m0 = paramvec.vector().array() #np.array([float(control.data()) for control in controls])
    for eps in epsilons:
        m_new = np.array(m0 + eps*d)
       
        #paramvec.assign(Constant(m_new))
        Jm = Jhat(m_new)
        no_grad_errors.append(abs(Jm - Jm0))
        grad_errors.append(abs(Jm - Jm0 - np.dot(DJm0, m_new - m0)))
       
    print "Errors without gradient"
    print no_grad_errors
    print "Convergence orders without gradient (should be 1)"
    print convergence_order(no_grad_errors)
   
    print "\nErrors with gradient"
    print grad_errors
    print "Convergence orders with gradient (should be 2)"
    con_ord = convergence_order(grad_errors)
    print con_ord
    if (np.array(con_ord) > 1.9).all():
        print "PASSED"
        return True
    else:
        print "FAILED"
        return False

def passive_taylor_test(rd, controls, Jm0):
    # from IPython import embed; embed()
    DJm0 = rd.derivative(forget=False)
    set_log_active(True)
    
    conv_rate = taylor_test(rd, controls, Jm0, DJm0[0])  
    print conv_rate
    assert abs(conv_rate -2) < 0.1, "Taylor test failed for passive phase"


def passive_test_functional(rd, paramvec, for_run, params):
    
    paramvec.assign(Constant(params["Material_parameters"].values()))
    rd_val = rd(paramvec)

    for_run_val = for_run(paramvec).func_value

    print "RD = ", rd_val
    print "For run = ", for_run_val

def test_passive(params, patient):
    
    logger.info(Text.blue("\nTest Passive Optimization"))

    logger.debug(pformat(params.to_dict()))


    params["phase"] = "passive_inflation"
    parameters["adjoint"]["test_derivative"] = True
    # parameters["adjoint"]["stop_annotating"] = True
    measurements, solver_parameters, p_lv, paramvec = \
      setup_simulation(params, patient)

    
    control, rd, for_run, forward_result = \
      run_passive_optimization_step(params, 
                                    patient, 
                                    solver_parameters, 
                                    measurements, 
                                    p_lv, paramvec)

    set_log_active(True)
    # replay_dolfin(tol=1e-12)
    # passive_test_functional(rd, paramvec, for_run, params)
    # print "RD=", rd(paramvec)
    # print "for_run=", forward_result.func_value
    passive_taylor_test(rd, control, forward_result.func_value)
    # Dump html visualization of the forward system
    # adj_html("passive_forward.html", "forward")

def get_args(phase = "", mode = ""):
    assert mode in MODES, "mode {} is not a legal mode".format(mode)
    
    if phase == "active":
        parser = make_activation_param_parser()

    elif phase == "passive":
        parser = make_material_param_parser()
        
    else:
        raise ValueError("Unknown phase {}".format(phase))

    args,_= parser.parse_known_args()
    args.initial_params = [2.1, 2.1, 2.5, 2.3]
    args.sim_file = "simulation.h5"
    args.synth_data = True
    args.mode = mode
    args.alpha = 0.5
    args.alpha_matparams = 0.5
    args.use_deintegrated_strains = False
    return args


def test_replay_dolfin_passive():
    patient = create_patient_data()
    args = get_args("passive", "replay_test")
    adj_reset()
    run_passive_optimization(args, patient)



def test_active():
    patient = create_patient_data()
    args = get_args("passive", "optimize")
    args.optimize_matparams = False
    adj_reset()
    # run_passive_optimization(args, patient)
    adj_reset()

    args = get_args("active", "replay_test")
    measurements, solver_parameters, p_lv, p_rv, paramvec = setup_simulation(args, patient)
    args.active_contraction_iteration_number = 0
    rd, gamma, gamma_previous = run_active_optimization_step(args, patient, solver_parameters, measurements, p_lv, p_rv)
    # gamma.assign(Constant(0.5))
    # gamma_arr = gather_broadcast(gamma.vector().array())
    # mpi_print(rd(gamma_arr))

    # h5filepath = rd.for_run.h5filepath
    # h5group =  ACTIVE_CONTRACTION_GROUP.format(args.alpha, args.active_contraction_iteration_number) + "/regional"
        
    # Write results
    # write_opt_results_to_h5(h5filepath, h5group, args, rd.ini_for_res,
				# rd.for_res, opt_gamma = rd.gamma)
    

    # args.active_contraction_iteration_number = 1
    # rd, gamma, gamma_previous = run_active_optimization_step(args, patient, solver_parameters, measurements, p_lv, p_rv)
    gamma.assign(Constant(0.1))
    gamma_arr = gather_broadcast(gamma.vector().array())

    r = rd(gamma_arr)
    mpi_print(replay_dolfin())

    dr = rd.derivative(forget=False)
    taylor_test(rd, gamma_arr, r, dr)
    




def test_functional_active():
    pass



def test_gradient_passive():
    patient = create_patient_data()
    args = get_args("passive", "test_gradient")
    adj_reset()
    run_passive_optimization(args, patient)


def test_gradient_active():
    pass

def run_optimization():
    pass
            
        

def create_patient_data():
    pressure = [0,1,2,3,2]
    passive_filling_duration = 3
    patient = IdializedPatient(pressure, passive_filling_duration, "test.h5")
    
    return patient
    

    
    



def main():
    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    params["Patient_parameters"]["patient"] = "test"
    params["Patient_parameters"]["patient_type"] = "test"
    params["sim_file"] = "data/test.h5"
    params["outdir"] = "data"

    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])
    test_passive(params,patient)


if __name__ == "__main__":
    main()

    # set_log_active(True)
    # set_log_level(DEBUG)
    # create_patient_data()
    #test_replay_dolfin_passive()
    #test_functional_passive()
    #test_gradient_passive()
    # test_active()
