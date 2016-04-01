from dolfin import *
from dolfin_adjoint import *
from campass.adjoint_contraction_args import *
from campass.setup_optimization import setup_adjoint_contraction_parameters, setup_general_parameters
import numpy as np
from numpy_mpi import *
from campass.utils import Text

def setup_params():
    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    params["Patient_parameters"]["patient"] = "test"
    params["Patient_parameters"]["patient_type"] = "test"
    params["sim_file"] = "data/test.h5"
    params["outdir"] = "data"
    set_log_active(True)

    logger.setLevel(DEBUG)

    return params


def passive_test_functional(rd, paramvec, for_run, params):
    
    # Pick two different parameter sets
    matparams = [params["Material_parameters"].values(), 
                 np.add(params["Material_parameters"].values(), 0.5)]

    for m in matparams:

        # Assign the material parameter parameters
        paramvec.assign(Constant(m))

        # Comput the reduced functionals
        rd_val = rd(paramvec)

        for_run_val = for_run(paramvec).func_value
        
        logger.info("Material parameters = {}".format(m))
        logger.info(Text.green("Reduced functional value = {}".format(rd_val)))
        logger.info(Text.green("Forward value = {}".format(for_run_val)))
        
        assert abs(rd_val-for_run_val) < 1e-4

def passive_taylor_test(Jhat, controls):#controls, Jm0):

    paramvec = controls.data()
    Jm0 = Jhat(paramvec)
    DJm0_fun = Jhat.derivative(forget=False)

    DJm0 = DJm0_fun[0].vector().array()
    m0 = paramvec.vector().array()
    d = np.array([1.0]*len(m0)) #perturbation direction
    grad_errors = []
    no_grad_errors = []
   
    epsilons = [0.05, 0.025, 0.0125]

    
    for eps in epsilons:
        m_new = np.array(m0 + eps*d)
       
        paramvec.assign(Constant(m_new))
        Jm = Jhat(paramvec)
        no_grad_errors.append(abs(Jm - Jm0))
        grad_errors.append(abs(Jm - Jm0 - np.dot(DJm0, m_new - m0)))
       
    logger.info("Errors without gradient: {}".format(no_grad_errors))
    logger.info("Convergence orders without gradient (should be 1)")
    logger.info("{}".format(convergence_order(no_grad_errors)))
   
    logger.info("\nErrors with gradient: {}".format(grad_errors))
    logger.info("Convergence orders with gradient (should be 2)")
    con_ord = convergence_order(grad_errors)
    logger.info("{}".format(con_ord))
    
    assert (np.array(con_ord) > 1.9).all()



def active_taylor_test(Jhat, m0_fun):
    m0 = gather_broadcast(m0_fun.vector().array())

    Jm0 = Jhat(m0)

    DJm0 = Jhat.derivative(forget=False)

    d = np.array([1.0]*len(m0)) #perturbation direction
    grad_errors = []
    no_grad_errors = []
   
    epsilons = [0.05, 0.025, 0.0125]

    
    for eps in epsilons:
        m_new = np.array(m0 + eps*d)
       
        Jm = Jhat(m_new)
        no_grad_errors.append(abs(Jm - Jm0))
        grad_errors.append(abs(Jm - Jm0 - np.dot(DJm0, m_new - m0)))
       
    logger.info("Errors without gradient: {}".format(no_grad_errors))
    logger.info("Convergence orders without gradient (should be 1)")
    logger.info("{}".format(convergence_order(no_grad_errors)))
   
    logger.info("\nErrors with gradient: {}".format(grad_errors))
    logger.info("Convergence orders with gradient (should be 2)")
    con_ord = convergence_order(grad_errors)
    logger.info("{}".format(con_ord))
    
    assert (np.array(con_ord) > 1.9).all()
