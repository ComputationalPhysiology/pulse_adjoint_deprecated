#!/usr/bin/env python
"""
This script implements that various opimtimization options
that can be used to solve the optimal control problem

.. math::

   \min_{m} \mathcal{J}(\mathbf{u},p,m)  

   \mathrm{subject}\;\mathrm{to:} \: \delta\Pi(\mathbf{u},p,m) = 0


**Example of usage**::

  # Suppose you allready have initialized you reduced functional (`rd`)
  # with the control parameter (`paramvec`)
  # Look at run_optimization.py to see how to do this. 

  # Initialize the paramters
  params = setup_application_parameters()
  
  # params["Optimization_parameters"]["opt_type"] = "pyOpt_slsqp"
  # params["Optimization_parameters"]["opt_type"] = "scipy_l-bfgs-b"
  params["Optimization_parameters"]["opt_type"] = "scipy_slsqp"

  # Create the optimal control problem
  oc_problem = OptimalControl()
  # Build the problem
  oc_problem.build_problem(params, rd, paramvec)
  # Solve the optimal control problem
  rd, opt_result = oc_problem.solve()

"""
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
import numpy as np
from dolfin import Timer
from adjoint_contraction_args import logger
from utils import print_line, print_head

try:
    import scipy
    from scipy.optimize import minimize as scipy_minimize
    from scipy.optimize import minimize_scalar as scipy_minimize_1d
    has_scipy = True
    from distutils.version import StrictVersion
    has_scipy016 = StrictVersion(scipy.version.version) >= StrictVersion ('0.16')
    
except:
    has_scipy = False
    has_scipy016 = False

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

try:
    import pyOpt
    has_pyOpt = True
except:
    has_pyOpt = False

opt_import = [has_scipy, has_moola, has_pyOpt, has_pyipopt]

from adjoint_contraction_args import *
from numpy_mpi import gather_broadcast, assign_to_vector

class MyCallBack(object):
    """pass a custom callback function

    This makes sure it's being used.
    """
    def __init__(self, rd, tol, max_iter):

        
        self.ncalls = 0
        self.rd = rd
        self.opt_funcvalues= []

        logger.info("\n"+"Starting optimization".center(100, "-"))
        logger.info("Scale: {}, \nDerivative Scale: {}".format(rd.scale,
                                                                 rd.derivative_scale))
        logger.info("Tolerace: {}, \nMaximum iterations: {}\n".format(tol, max_iter))
        logger.info(print_head(rd.for_res))

    def __call__(self, x):
       
        self.ncalls += 1
        
        grad_norm = None if len(self.rd.grad_norm_scaled) == 0 \
                    else self.rd.grad_norm_scaled[-1]

        func_value = self.rd.for_res["func_value"]
        self.opt_funcvalues.append(func_value)
        self.rd.opt_funcvalues = self.opt_funcvalues
        
        logger.info(print_line(self.rd.for_res, self.ncalls,
                               grad_norm, func_value))
        
       

def minimize_1d(f, x0, **kwargs):
    """Minimize functional with one variable using the 
    brent algorithm from scpiy.

    :param f: Objective functional
    :type f: :py:class:`setup_optimization.MyReducedFuntional`
    :param float x0: initial guess
    :returns: Scipy results from the opimization
    :rtype: 

    """
    

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

def get_ipopt_options(rd, lb, ub, tol, max_iter, **kwargs):
    """Get options for IPOPT module (interior point algorithm)

    See `<https://projects.coin-or.org/Ipopt>`

    :param rd: The reduced functional
    :param list lb: Lower bound on the control
    :param list ub: Upper bound on the control
    :param tol: Tolerance
    :param max_iter: Maximum number of iterations
    :returns: The optimization solver and the options
    :rtype: dict

    """
    
    ncontrols = len(ub)
    nconstraints = 0
    empty = np.array([], dtype=float)
    clb = empty
    cub = empty
    constraints_nnz = nconstraints * ncontrols
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
        
    J = rd.__call__
    dJ = rd.derivative

    nlp = pyipopt.create(ncontrols,         # length of control vector
                         lb,                # lower bounds on control vector
                         ub,                # upper bounds on control vector
                         0,                 # number of constraints
                         clb,               # lower bounds on constraints,
                         cub,               # upper bounds on constraints,
                         0,                 # number of nonzeros in the constraint Jacobian
                         0,                 # number of nonzeros in the Hessian
                         J,                 # to evaluate the functional
                         dJ,                # to evaluate the gradient
                         fun_g,             # to evaluate the constraints
                         jac_g)             # to evaluate the constraint Jacobian

    pyipopt.set_loglevel(1)
    return nlp


def get_moola_options(method, rd, lb, ub, tol, max_iter, **kwargs):
    """Get options for moola module.

    See `<https://github.com/funsim/moola>`

    :param str method: Which optimization algorithm
    :param rd: The reduced functional
    :param list lb: Lower bound on the control
    :param list ub: Upper bound on the control
    :param tol: Tolerance
    :param max_iter: Maximum number of iterations
    :returns: The optimization solver and the options
    :rtype: dict

    .. note::
    
       This is not working

    """
    
    # problem = MoolaOptimizationProblem(rd)
                
    # paramvec_moola = moola.DolfinPrimalVector(paramvec)
    # solver = moola.NewtonCG(problem, paramvec_moola, options={'gtol': 1e-9,
    #                                                           'maxiter': 20, 
    #                                                           'display': 3, 
    #                                                           'ncg_hesstol': 0})
    
                
    # solver = moola.BFGS(problem, paramvec_moola, options={'jtol': 0,
    #                                                       'gtol': 1e-9,
    #                                                       'Hinit': "default",
    #                                                       'maxiter': 100,
    #                                                     'mem_lim': 10})
    
    # solver = moola.NonLinearCG(problem, paramvec_moola, options={'jtol': 0,
    #                                                          'gtol': 1e-9,
    #                                                          'Hinit': "default",
    #                                                          'maxiter': 100,
    #                                                          'mem_lim': 10})
    
    raise NotImplementedError

    
def get_scipy_options(method, rd, lb, ub, tol, max_iter, **kwargs):
    """Get options for scipy module

    See `<https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.html>`

    :param str method: Which optimization algorithm 'LBFGS' or 'SLSQP'
    :param rd: The reduced functional
    :param list lb: Lower bound on the control
    :param list ub: Upper bound on the control
    :param tol: Tolerance
    :param max_iter: Maximum number of iterations
    :returns: The optimization solver and the options
    :rtype: dict

    """

    
    def lowerbound_constraint(m):
        return m - lb
            
    def upperbound_constraint(m):
        return ub - m



    cons = ({"type": "ineq", "fun": lowerbound_constraint},
            {"type": "ineq", "fun": upperbound_constraint})                
                

    if not has_scipy016 and method == "slsqp":
        callback = None
    else:
        callback = MyCallBack(rd, tol, max_iter)

    
    options = {"method": method,
               "jac": rd.derivative,
               "tol":tol,
               "callback": callback,
               "options": {"disp": kwargs.pop("disp",False),
                           "iprint": kwargs.pop("iprint",2),
                           "ftol": tol,
                           "maxiter":max_iter}}

    if method == "slsqp":
        options["constraints"] = cons
    else:
        options["bounds"] = zip(lb,ub)
        
    return options

def get_pyOpt_options(method, rd, lb, ub, tol, max_iter, **kwargs):
    """Get options for pyOpt module

    See `<http://www.pyopt.org>`

    :param str method: Which optimization algorithm `not working` SLSQP will be chosen.
    :param rd: The reduced functional
    :param list lb: Lower bound on the control
    :param list ub: Upper bound on the control
    :param tol: Tolerance
    :param max_iter: Maximum number of iterations
    :returns: The optimization solver and the options
    :rtype: dict

    """
    

    def obj(x):

        
        f, fail = rd(x, True)

        g = []
        
        return f,g,fail
    

    def grad(x,f,g):
        fail = False
        try:
            dj = rd.derivative()
        except:
            fail = True

        # Contraints gradient
        gJac = np.zeros(len(x))
      
        # logger.info("j = %f\t\t|dJ| = %f" % (f[0], np.linalg.norm(dj)))
        return np.array([dj]), gJac, fail


    # Create problem
    opt_prob = pyOpt.Optimization('Problem',obj)

    # Assign objective
    opt_prob.addObj('J')

    # Assign design variables (bounds)
    opt_prob.addVarGroup("variables", kwargs["nvar"], type='c',
                         value=kwargs["m"], lower=lb, upper=ub)


    opt = pyOpt.pySLSQP.SLSQP()
    opt.setOption("ACC", tol)
    opt.setOption("MAXIT", max_iter)
    opt.setOption("IPRINT", -1)
    opt.setOption("IFILE", "")


    options = {"opt_problem":opt_prob,
               "sens_type": grad}

    return opt, options
    

class OptimalControl(object):
    """
    A class used for solving an optimal control problem

    """
    def build_problem(self, params, rd, paramvec):
        """Build optimal control problem

        :param dict params: Application parameter
        :param rd: The reduced functional
        :param paramvec: Control parameter
       
        """
        

        msg = "No supported optimization module installed"
        assert any(opt_import), msg


        opt_params = params["Optimization_parmeteres"].to_dict()

        x = gather_broadcast(paramvec.vector().array())
        nvar = len(x)
        self.paramvec = paramvec
        self.x = x
        self.rd = rd

        if params["phase"] == PHASES[0]:
            
            lb = np.array([opt_params["matparams_min"]]*nvar)
            ub = np.array([opt_params["matparams_max"]]*nvar)
                
            tol = opt_params["passive_opt_tol"]
            max_iter = opt_params["passive_maxiter"]

        else:

            lb = np.array([opt_params["gamma_min"]]*nvar)
            ub = np.array([opt_params["gamma_max"]]*nvar)
                

            tol= opt_params["active_opt_tol"]
            max_iter = opt_params["active_maxiter"]

        self.tol = tol

        if nvar == 1:

            self.options = {"method": opt_params["method_1d"],
                            "bounds":zip(lb,ub)[0],
                            "tol":tol,
                            "options": {"maxiter":max_iter}}

            self.oneD = True
            self.opt_type = "scipy_brent"
            
        else:

            self.oneD = False
            
            opt_params["nvar"]= nvar
            opt_params["m"]= x
            
            self.opt_type = opt_params.pop("opt_type", "scipy_slsqp")
            self._get_options(lb, ub, tol, max_iter, **opt_params)

            
        

    def _get_options(self, lb, ub, tol, max_iter, **kwargs):

        module, method = self.opt_type.split("_")

        if module == "scipy":
            assert has_scipy, "Scipy not installed"
            self.options = get_scipy_options(method, self.rd, lb, ub, tol, max_iter, **kwargs)

        elif module == "moola":
            assert has_moola, "Moola not installed"
            self.solver = get_moola_options(method, self.rd, lb, ub, tol, max_iter, **kwargs)

        elif module == "pyOpt":
            assert has_pyOpt, "pyOpt not installed"
            self.problem, self.options = get_pyOpt_options(self.rd, lb, ub, tol, max_iter, **kwargs)

        elif module == "ipopt":
            assert has_pyipopt, "IPOPT not installed"
            self.solver = get_ipopt_options(method, self.rd, lb, ub, tol, max_iter, **kwargs)

        else:
            msg = ("Unknown optimizatin type {}. "
                   "Define the optimization type as 'module-method', "
                   "where module is e.g scipy, pyOpt and methos is "
                   "eg slsqp.")
            raise ValueError(msg)
                              

                      
                      
    
    def solve(self):
        """
        Solve optmal control problem

        """

        msg = "You need to build the problem before solving it"
        assert hasattr(self, "opt_type"), msg
            

        module, method = self.opt_type.split("_")

        t = Timer()
        t.start()

        if self.oneD:
            res = minimize_1d(self.rd, paramvec_arr[0], **self.options)
            x = res["x"]
            
        else:
        
            if module == "scipy":
                # from IPython import embed; embed()
                # exit()
                res = scipy_minimize(self.rd,self.x, **self.options)
                x = res["x"]
                
            elif module == "pyOpt":
            
                obj, x, d = self.problem(**self.options)
                
            elif module == "moola":

                sol = self.solver.solve()
                x = sol['control'].data
                
            elif module == "ipopt":
                x = self.solver.solve(self.x)
                
            else:
                msg = ("Unknown optimizatin type {}. "
                       "Define the optimization type as 'module-method', "
                       "where module is e.g scipy, pyOpt and methos is "
                       "eg slsqp.")
                raise ValueError(msg)

            

        run_time = t.stop()
    
        opt_result = {}

        opt_result["x"] = x
        opt_result["nfev"] = self.rd.iter
        opt_result["nit"] = self.rd.iter
        opt_result["njev"] = self.rd.nr_der_calls
        opt_result["ncrash"] = self.rd.nr_crashes
        opt_result["run_time"] = run_time
        opt_result["controls"] = self.rd.controls_lst
        opt_result["func_vals"] = self.rd.func_values_lst
        opt_result["forward_times"] = self.rd.forward_times
        opt_result["backward_times"] = self.rd.backward_times
        opt_result["grad_norm"] = self.rd.grad_norm
              
    
        return self.rd, opt_result
     


