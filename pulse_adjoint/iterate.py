from dolfinimport import *
import numpy as np
from adjoint_contraction_args import logger
from numpy_mpi import *
from lvsolver import SolverDidNotConverge
import operator as op

MAX_GAMMA_STEP = 0.05
MAX_PRESSURE_STEP = 0.4
MAX_CRASH = 10

    

def get_diff(current, target, control):

    if control == "gamma":
        diff = Vector(target.vector())
        diff.axpy(-1.0, current.vector())
        
    elif control == "pressure":
        diff = np.subtract(target, current)

    else:
        raise ValueError("Unknown control mode {}".format(control_mode))

    return diff

def get_current_control_value(solver, p_expr, control):

    if control == "gamma":
        return solver.parameters["material"].gamma.copy(True)
    
    elif control == "pressure":
        if p_expr.has_key("p_rv"):
            return (p_expr["p_lv"].t, p_expr["p_rv"].t)
        else:
            return p_expr["p_lv"].t

def assign_new_control(solver, p_expr, control, new_control):

    if control == "gamma":
        solver.parameters["material"].gamma.assign(new_control)
        
    elif control == "pressure":
        if p_expr.has_key("p_rv"):
            p_expr["p_lv"].t = new_control[0]
            p_expr["p_rv"].t = new_control[1]
        else:
            
            p_expr["p_lv"].t = new_control
    else:
        raise ValueError("Unknown control mode {}".format(control_mode))



def check_target_reached(solver, p_expr, control, target):

    current = get_current_control_value(solver, p_expr, control)
    diff = get_diff(current, target, control)
     
    if control == "gamma":
        max_diff = norm(diff, 'linf') 
    
    elif control == "pressure":
        max_diff = np.max(abs(diff))

    
    reached = max_diff < DOLFIN_EPS
    if reached:
        logger.info("Check target reached: YES!")
    else:
        logger.info("Check target reached: NO")
        logger.info("Maximum difference: {:.3e}".format(max_diff))
        
    return reached

def get_initial_step(solver, p_expr, control, target):
    
    current = get_current_control_value(solver, p_expr, control)
    diff = get_diff(current, target, control)
    
    if control == "gamma":
        
        cur_arr = gather_broadcast(current.vector().array())
        max_diff = norm(diff, 'linf') + 0.1*abs(cur_arr.max() - cur_arr.min())

        nsteps = np.ceil(float(max_diff)/MAX_GAMMA_STEP) + 1
        step = Function(current.function_space(), name = "gamma_step")
        step.vector().zero()
        step.vector().axpy(1.0/float(nsteps), diff)
        
    elif control == "pressure":
        max_diff = abs(np.max(diff))
        nsteps = float(max_diff) / MAX_PRESSURE_STEP + 1
        step = diff/float(nsteps)

    logger.debug("Intial number of steps: {}".format(nsteps))
    return step

    
def step_too_large(current, target, step, control):

    if control == "gamma":
        

        diff_before = Vector(current.vector())
        diff_before.axpy(-1.0, target.vector())
        
        
        diff_after= Vector(current.vector())
        diff_after.axpy(1.0, step.vector())
        diff_after.axpy(-1.0, target.vector())


        return not all(np.sign(diff_before.array()) == \
                       np.sign(diff_after.array()))

        
        
    elif control == "pressure":

        if isinstance(target, (float, int)):
            comp = op.gt if current < target else op.lt
            return comp(current + step, target)
        else:
            assert hasattr(target,"__len__")

            too_large = []
            for (c,t, s) in zip(current, target, step):
                comp = op.gt if c < t else op.lt
                too_large.append(comp(c+s, t))
                
            return all(too_large)
        

def change_step_size(step, factor, control):

    if control == "gamma":
        step.vector().axpy(factor-1.0, Vector(step.vector()))
        
    elif control == "pressure":
        step = np.multiply(factor, step)

    return step


def print_control(control):


    def print_arr(arr):

        if len(arr) == 2:
            # This has to be (LV, RV)
            logger.info("\t{:>6}\t{:>6}".format("LV", "RV"))
            logger.info("\t{:>6.2f}\t{:>6.2f}".format(arr[0],
                                                      arr[1]))
            
        elif len(arr) == 3:
            # This has to be (LV, Septum, RV)
            logger.info("\t{:>6}\t{:>6}\t{:>6}".format("LV", "SEPT", "RV"))
            logger.info("\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}".format(arr[0],
                                                                arr[1],
                                                                arr[2]))
        else:
            # Print min, mean and max
            logger.info("\t{:>6}\t{:>6}\t{:>6}".format("Min", "Mean", "Max"))
            logger.info("\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}".format(np.min(arr),
                                                                np.mean(arr),
                                                                np.max(arr)))        
    
    if isinstance(control, (float, int)):
        logger.info("\t{:>6.3f}".format(control))
    elif isinstance(control, (dolfin.Function, dolfin_adjoint.Function)):
        arr = gather_broadcast(control.vector().array())
        logger.info("\t{:>6}\t{:>6}\t{:>6}".format("Min", "Mean", "Max"))
        logger.info("\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}".format(np.min(arr),
                                                            np.mean(arr),
                                                            np.max(arr)))
    elif isinstance(control, (dolfin.GenericVector, dolfin.Vector)):
        arr = gather_broadcast(control.array())
        print_arr(arr)
        
    elif isinstance(control, (tuple, np.ndarray, list)):
        print_arr(control)

def get_delta(new_control, c0, c1):

    if isinstance(new_control, (int, float)):
        return  (new_control - c0)/float(c1 - c0)

    elif isinstance(new_control, (tuple, np.ndarray, list)):
        return  (new_control[0] - c0[0])/float(c1[0] - c0[0])

    elif isinstance(new_control, (dolfin.GenericVector, dolfin.Vector)):
        new_control_arr = gather_broadcast(new_control.array())
        c0_arr = gather_broadcast(c0.array())
        c1_arr = gather_broadcast(c1.array())
        return  (new_control_arr[0] - c0_arr[0])/float(c1_arr[0] - c0_arr[0])

    elif isinstance(new_control, (dolfin.Function, dolfin_adjoint.Function)):
        new_control_arr = gather_broadcast(new_control.vector().array())
        c0_arr = gather_broadcast(c0.vector().array())
        c1_arr = gather_broadcast(c1.vector().array())
        return  (new_control_arr[0] - c0_arr[0])/float(c1_arr[0] - c0_arr[0])
        
        
def iterate(solver, target, control = "gamma", p_expr=None,
            continuation = True, max_adapt_iter = 8, adapt_step=True):
    """
    

    Parameters
    ----------

    solver (LVSolver)
        The solver
    target (dolfin.Function or tuple or float)
        The target value. Typically a float if target is LVP, a tuple
        if target is (LVP, RVP) and a function if target is gamma.
    control (str)
        Control mode, so far either 'pressure' or 'gamma'
    p_expr (dict)
        A dictionary with expression for the pressure and keys
        'p_lv' (and 'p_rv' if BiV)
    continuation (bool)
        Apply continuation for better guess for newton solver
        Note: Replay test seems to fail when continuation is True, 
        but taylor test passes
    max_adapt_iter (int)
        If number of iterations is less than this number and adapt_step=True,
        then adapt control step
    adapt_step (bool)
        Adapt / increase step size when sucessful iterations are achevied. 

    
    """

    if control == "pressure":
        assert p_expr is not None, "provide the pressure"
        assert isinstance(p_expr, dict), "p_expr should be a dictionray"
        assert p_expr.has_key("p_lv"), "p_expr do not have the key 'p_lv'"
        
        

    target_reached = check_target_reached(solver, p_expr, control, target)
    logger.info("\nIterate Control: {}".format(control))

    
    step = get_initial_step(solver, p_expr, control, target)

    value = get_current_control_value(solver, p_expr, control)
    logger.info("Current value")
    print_control(value)
    
    control_values  = [value]

    
    prev_states = [solver.get_state().copy(True)]

    ncrashes = 0
    while not target_reached:

        if ncrashes > MAX_CRASH:
            raise SolverDidNotConverge

        control_value_old = control_values[-1]
        state_old = prev_states[-1]

        first_step = len(prev_states) < 2
        

        # Check if we are close
        if step_too_large(control_value_old, target, step, control):
            logger.info("Change step size for final iteration")
            # Change step size so that target is reached in the next iteration
            if control == "gamma":
                step = Function(target.function_space(), name = "gamma_step")
                step.vector().axpy(1.0, target.vector())
                step.vector().axpy(-1.0, control_value_old.vector())
            elif control == "pressure":
                step = target-control_value_old

        
        
        new_control = get_current_control_value(solver, p_expr, control)
        if control == "gamma":
            new_control.vector().axpy(1.0, step.vector())
        elif control == "pressure":
            new_control += step
    
        assign_new_control(solver, p_expr, control, new_control)        
        logger.info("\nTry new {}".format(control))
        print_control(new_control)


        # Prediction step (Make a better guess for newtons method)
        # Assuming state depends continuously on the control
        if not first_step and continuation:
            c0, c1 = control_values[-2:]
            s0, s1 = prev_states

            delta = get_delta(new_control, c0, c1)
            

            solver.get_state().vector().zero()
            solver.get_state().vector().axpy(1.0-delta, s0.vector())
            solver.get_state().vector().axpy(delta, s1.vector())

        
        
        try:
            nliter, nlconv = solver.solve()
        except:
            logger.info("\nNOT CONVERGING")
            logger.info("Reduce control step")

            if control == "gamma":
                new_control.vector().axpy(-1.0, step.vector())
            elif control == "pressure":
                new_control -= step
     
            # Assign old state
            logger.debug("Assign old state")
            # solver.reinit(state_old)
            solver.get_state().vector().zero()
            solver.get_state().vector().axpy(1.0, state_old.vector())

            # Assign old control value
            logger.debug("Assign old control")
            assign_new_control(solver, p_expr, control, new_control)
            # Reduce step size
            step = change_step_size(step, 0.5, control)
            
            continue
        
        else:
            logger.info("\nSUCCESFULL STEP:")

            target_reached = check_target_reached(solver, p_expr, control, target)

            if not target_reached:

                if nliter < max_adapt_iter and adapt_step:
                    logger.info("Adapt step size. New step size:")
                    step = change_step_size(step, 2.0, control)
                    print_control(step)
                
                control_values.append(new_control)

                if first_step:
                    prev_states.append(solver.get_state().copy(True))
                else:
                
                    # Switch place of the state vectors
                    prev_states = [prev_states[-1], prev_states[0]]

                    # Inplace update of last state values
                    prev_states[-1].vector().zero()
                    prev_states[-1].vector().axpy(1.0, solver.get_state().vector())

    return control_values, prev_states
        
        


def _get_solver(biv = False):

    from setup_parameters import setup_general_parameters, setup_application_parameters
    from utils import QuadratureSpace
    from material import HolzapfelOgden
    from lvsolver import LVSolver
    from setup_optimization import RegionalParameter
    
    setup_general_parameters()
    params = setup_application_parameters()

    from patient_data import LVTestPatient, BiVTestPatient

    if biv:
        patient = BiVTestPatient()
    else:
        patient = LVTestPatient()
    

    mesh = patient.mesh
    ffun = patient.ffun
    N = FacetNormal(mesh)
    # element_type = "mini"
    element_type = "taylor_hood"


    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W.sub(0)
        if element_type == "mini":
            P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
            B  = VectorFunctionSpace(mesh, "Bubble", 4)
            V1 = P1+B
            zero = project(Constant((0, 0, 0)), V1)
        else:
            zero = Constant((0,0,0))
            
        no_base_x_tran_bc = DirichletBC(V, zero, patient.BASE)
        
        # V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        # no_base_x_tran_bc = DirichletBC(V.sub(0), 0, patient.BASE)
        return no_base_x_tran_bc


    # Fibers
    V_f = QuadratureSpace(mesh, 4)
    # Unit field in x-direction
    f0 = patient.fiber
 
    # Contraction parameter
    # gamma = Function(FunctionSpace(mesh, "R", 0))
    gamma = RegionalParameter(patient.sfun)

    # Pressure
    p_lv = Expression("t", t = 0)
    if biv:
        p_rv = Expression("t", t =0)
        pressure = {"p_lv":p_lv, "p_rv":p_rv}
        neumann_bc = [[p_lv, patient.markers["ENDO_LV"][0]],
                     [p_rv, patient.markers["ENDO_RV"][0]]]
    else:
        neumann_bc = [[p_lv, patient.markers["ENDO"][0]]]
        pressure = {"p_lv":p_lv}
        

    # Spring
    spring = Constant(0.0)

    matparams = {"a":1.0, "a_f":1.0, 
                 "b":5.0, "b_f":5.0}
    # Set up material model
    # material = HolzapfelOgden(f0, gamma, active_model = "active_stress")
    material = HolzapfelOgden(f0, gamma, matparams, active_model = "active_strain")
    # material = NeoHookean(f0, gamma, active_model = "active_stress")

    # Solver parameters
    solver_parameters = {"snes_solver":{}}
    solver_parameters["nonlinear_solver"] = "snes"
    solver_parameters["snes_solver"]["method"] = "newtontr"
    solver_parameters["snes_solver"]["maximum_iterations"] = 8
    solver_parameters["snes_solver"]["absolute_tolerance"] = 1e-5
    solver_parameters["snes_solver"]["relative_tolerance"] = 1e-10
    solver_parameters["snes_solver"]["linear_solver"] = "lu"

    # Create parameters for the solver
    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             # "state_space": "P_2:P_1",
             "elements": element_type, 
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":neumann_bc,
                   "robin":[[spring, patient.BASE]]},
             "solve":solver_parameters}

    parameters["adjoint"]["stop_annotating"] = True
    solver = LVSolver(params)
    

    return solver, gamma, pressure

        
    
def test_stepping():
    from adjoint_contraction_args import logger
    logger.setLevel(10)
    
    solver, gamma, pressure = _get_solver(biv = True)
    # solver, gamma, pressure = _get_solver(biv = False)
    logger.info("\nIntial solve")
    solver.solve()

    
    target_gamma = gamma.copy(True)

    val = 0.2
    zero = Constant(val) if gamma.value_size() == 1 \
           else Constant(np.linspace(0, val, gamma.value_size()))
    
    target_gamma.assign(zero)

    # target = gamma.copy(True)
    # target.assign(Constant(0.2))

    # current = gamma.copy(True)
    # current.assign(Constant(0.15))

    # iterate(solver, target_gamma, "gamma", pressure)
    # iterate(solver, 2.0, "pressure", pressure)
    iterate(solver, (2.0, 1.0), "pressure", pressure)


if __name__ == "__main__":
    test_stepping()
