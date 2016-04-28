#!/usr/bin/env python
from dolfin import *
from dolfin_adjoint import *
from compressibility import Compressibility

import collections, math, numpy as np


def subplus(x):
    return conditional(ge(x, 0.0), x, 0.0)

def heaviside(x):
    return conditional(ge(x, 0.0), 1.0, 0.0)


class HolzapfelOgden(object):
    def __init__(self, f0, gamma = None, params = None, active_model = "active_strain"):

        self.f0 = f0
        self.gamma = Constant(0, name="gamma") if gamma is None else gamma

        
        if params is None:
            params = self.default_parameters()

        for k,v in params.iteritems():
            setattr(self, k, v)

        self._active_model = 'active_strain'
    
    def default_parameters(self):
        return {"a":0.291, "a_f":2.582, 
                "b":5.0, "b_f":5.0}


    def W_1(self, I_1, diff=0):
        """
        Isotropic contribution.
        """
        # from IPython import embed; embed()
        a = self.a
        b = self.b
        # if float(a) < DOLFIN_EPS:
        #     return 0
        # elif float(b) < DOLFIN_EPS:
        #     if diff == 0:
        #         return a / 2.0 * (I_1 - 3)
        #     elif diff == 1:
        #         return a / 2.0
        #     elif diff == 2:
        #         return 0
        # else:
        if diff == 0:
            return a/(2.0*b) * (exp(b*(I_1 - 3)) - 1)
        elif diff == 1:
            return a/2.0 * exp(b*(I_1 - 3))
        elif diff == 2:
            return a*b/2.0 * exp(b * (I_1 - 3))

    def W_4(self, I_4, diff=0):
        """
        Anisotropic contribution.
        """
        a = self.a_f

        b = self.b_f

        if I_4 == 0:
            return 0

        # if float(a) < DOLFIN_EPS:
        #     return 0.0
        # elif float(b) < DOLFIN_EPS:
        #     if diff == 0:
        #         return a/2.0 * heaviside(I_4 - 1.0) * pow(I_4 - 1.0, 2)
        #     elif diff == 1:
        #         return a * subplus(I_4 - 1)
        #     elif diff == 2:
        #         return heaviside(I_4 - 1)
        # else:
        if diff == 0:
            return a/(2.0*b) * heaviside(I_4 - 1) * (exp(b*pow(I_4 - 1, 2)) - 1)
        elif diff == 1:
            return a * subplus(I_4 - 1) \
                     * exp(b * pow(I_4 - 1, 2))
        elif diff == 2:
            return a * heaviside(I_4 - 1) \
                     * (1 + 2.0 * b * pow(I_4 - 1, 2)) \
                     * exp(b * pow(I_4 - 1, 2))


    def strain_energy(self, F, p=None, gamma=None):
        """
        Total strain-energy density function.
        """

        C = F.T * F
        J = det(F)
        Jm23 = pow(J, -float(2)/3)

        # Usage of fibers or sheets
        f0 = self.f0
        
        # Activation
        gamma = self.gamma

        # Invariants
        I1  = Jm23 * tr(C)
        I4f =  Jm23 * inner(C*f0, f0) 

        

        # Active stress model
        if self._active_model == 'active_stress':
            W1   = self.W_1(I1)
            W4f  = self.W_4(I4f)
            
            Wactive = gamma * I4f
            W = W1 + W4f + Wactive 

        # Active strain model
        elif self._active_model == 'active_strain':
            mgamma = 1 - gamma
            I1e   = mgamma * I1 + (1/mgamma**2 - mgamma) * I4f
            I4fe  = 1/mgamma**2 * I4f
            
            
            W1   = self.W_1(I1e)
            W4f  = self.W_4(I4fe)
            
            W = W1 + W4f
        else:
            raise NotImplementedError("The active model '{}' is "\
                                      "not implemented.".format(\
                                          self._active_model))

        return W





def set_ffc_params():
    #Fast math does not confirm to floating point standard.
    #It is on now but should be removed if there is any doubt about it's effects.

    flags = ["-O3", "-ffast-math", "-march=native"]
    
    parameters["form_compiler"]["quadrature_degree"] = 4
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

class LVSolver(collections.Iterator):
    """
    A solver that solves the quasi-static Holzapfel and Ogden Elasticity
    equations. Can be used with a time loop by calling .next().
    Or can be used directly by calling .solve().
    """
    
    def __init__(self, parameters):        

        self.compressibility = Compressibility.Incompressible(parameters)
            
        self.W = self.compressibility.W
        self.w = self.compressibility.w
        self.w_test = self.compressibility.w_test
        
        if hasattr(parameters["bc"]["dirichlet"], '__call__'):
            self.bcs = parameters["bc"]["dirichlet"](self.W)
        else:
            self.bcs = self.make_dirichlet_bcs(parameters)
        
        
        self.parameters = parameters
        

        self.R, self.P = self.make_variational_form(parameters)
        self.J = derivative(self.R, self.w)
       
        self.initial_step = True        
        
    def get_displacement(self, name = "displacement", annotate = True):
        return self.compressibility.get_displacement(name, annotate)
    
    
        
    def make_dirichlet_bcs(self, p):
        bcs = []
        D = self.compressibility.get_displacement_space()
        for bc_spec in p["bc"]["dirichlet"]:
            val, marker = bc_spec
            if type(marker) == int:
                args = [D, val, p["facet_function"], marker]
            else:
                args = [D, val, marker]
            bcs.append(DirichletBC(*args))
        return bcs            
    
    def next(self):
        if self.initial_step:
            self.initial_step = False
            return self.w
        
        elif self.t < self.parameters["time"]["end"]:
            should_return = False
            crashed = False    

            while (not should_return) or crashed:
                next_dt, should_return = self._set_next_dt(self.dt)
                self.w, crashed = self._solve_with_step_shortening(next_dt)
            logger.debug("{}Returning Solution t = {}{}".format(_OKBLUE, self.t, _ENDC))
            return self.w
        else:
            raise StopIteration

    def _set_next_dt(self, dt):
        should_return = False
        in_interval = self.solution_times[(self.solution_times > self.t + 1.0e-16) & (self.solution_times <= self.t + dt)]            
        if in_interval.any():
            in_interval.sort()
            dt = in_interval[0] - self.t
            should_return = True

        if self.t + dt > self.parameters["time"]["end"] and self.t < self.parameters["time"]["end"]:
            dt = self.parameters["time"]["end"] - self.t
            should_return = True
        return dt, should_return

    def solve(self):
        
        
        solve(self.R == 0,
                self.w,
                self.bcs,
                J = self.J,
                solver_parameters = self.parameters["solve"],
                annotate = False)
    
       
        if not parameters["adjoint"]["stop_annotating"]:
            solve(self.R == 0,
                  self.w,
                  self.bcs,
                  J = self.J,
                  solver_parameters = self.parameters["solve"],
                  annotate = True)
        return self.w

    def make_variational_form(self, params):

        material = self.parameters["material"]

        # Displacement
        u = self.compressibility.get_displacement_variable()
        # Deformation gradient
        F = grad(u) + Identity(3)
        self.F = variable(F)
        J = det(self.F)

        # Isochoric Right Cauchy Green tensor
        Cbar = J**(-2.0/3.0)*self.F.T*self.F
        
        # Internal energy
        pi_int = material.strain_energy(F) + self.compressibility(J)
        
        Pdx, P = derivative(pi_int*dx, self.w, self.w_test), diff(pi_int, self.F)
        R = self.add_external_work(Pdx, u, F, J, params)                
        return R, P


    def add_external_work(self, pi_int, u, F, J, params):
        v = self.compressibility.u_test
        if params.has_key("facet_function"):
            ds = Measure("exterior_facet", subdomain_data = params["facet_function"])
        
        if params["bc"].has_key("Pressure"):
            N = params["facet_normal"]
            
            for pressure_bc in params["bc"]["Pressure"]:
                p, marker = pressure_bc
                pi_int += inner(J*p*dot(inv(F).T, N), v)*ds(marker)
        
        if params["bc"].has_key("Robin"):
            for robin_bc in params["bc"]["Robin"]:
                val, marker = robin_bc
                pi_int += -inner(val*u, v)*ds(marker)
        
        if params.has_key("body_force"):
            pi_int += -inner(params["body_force"], v)*dx
        return pi_int

