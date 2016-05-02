#!/usr/bin/env python
from dolfin import *
from dolfin_adjoint import *
from compressibility import get_compressibility
from adjoint_contraction_args import logger

import collections, math, numpy as np


def subplus(x):
    return conditional(ge(x, 0.0), x, 0.0)

def heaviside(x):
    return conditional(ge(x, 0.0), 1.0, 0.0)


class HolzapfelOgden(object):
    def __init__(self, f0, gamma = None, params = None, active_model = "active_strain"):

        assert active_model in ["active_strain", "active_stress"]
        self.f0 = f0
        self.gamma = Constant(0, name="gamma") if gamma is None else gamma

        
        if params is None:
            params = self.default_parameters()

        for k,v in params.iteritems():
            setattr(self, k, v)

        self._active_model = active_model
    
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

    def I1(self, F):
        """
        First Isotropic invariant
        """
        C = F.T * F
        J = det(F)
        Jm23 = pow(J, -float(2)/3)
        return  Jm23 * tr(C)

    def I4f(self, F):
        """
        Quasi invariant in fiber direction
        """
        C = F.T * F
        J = det(F)
        Jm23 = pow(J, -float(2)/3)
        return Jm23 * inner(C*self.f0, self.f0) 


    def strain_energy(self, F, p=None, gamma=None):
        """
        Strain-energy density function.
        """

        # Activation
        gamma = self.gamma

        # Invariants
        I1  = self.I1(F)
        I4f =  self.I4f(F)

        
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


class LVSolver(object):
    
    def __init__(self, params):        

        for k in ["mesh", "facet_function", "material", "bc"]:
            assert params.has_key(k), \
              "{} need to be in solver_parameters".format(k)

        
        self.parameters = params

        # Update solver parameters
        prm = self.default_solver_parameters(use_snes=True, iterative_solver=False)
        
        for k, v in params["solve"].iteritems():
            if isinstance(params["solve"][k], dict):
                for k_sub, v_sub in params["solve"][k].iteritems():
                    prm[k][k_sub]= v_sub

            else:
                prm[k] = v
            
        self.parameters["solve"] = prm
        
        
        self._init_spaces()
        self._init_forms()

    def default_solver_parameters(self, use_snes=True, iterative_solver=False):

        nsolver = "snes_solver" if use_snes else "newton_solver"

        prm = {"nonlinear_solver": "snes", "snes_solver":{}} if use_snes else {"nonlinear_solver": "newton", "newton_solver":{}}

        prm[nsolver]['absolute_tolerance'] = 1E-5
        prm[nsolver]['relative_tolerance'] = 1E-5
        prm[nsolver]['maximum_iterations'] = 8
        # prm[nsolver]['relaxation_parameter'] = 1.0
        prm[nsolver]['linear_solver'] = 'lu'
        prm[nsolver]['error_on_nonconvergence'] = True
        prm[nsolver]['report'] = True if logger.level < INFO else False
        if iterative_solver:
            prm[nsolver]['linear_solver'] = 'gmres'
            prm[nsolver]['preconditioner'] = 'ilu'

            prm[nsolver]['krylov_solver'] = {}
            prm[nsolver]['krylov_solver']['absolute_tolerance'] = 1E-9
            prm[nsolver]['krylov_solver']['relative_tolerance'] = 1E-7
            prm[nsolver]['krylov_solver']['maximum_iterations'] = 1000
            prm[nsolver]['krylov_solver']['monitor_convergence'] = False
            prm[nsolver]['krylov_solver']['nonzero_initial_guess'] = False

            prm[nsolver]['krylov_solver']['gmres'] = {}
            prm[nsolver]['krylov_solver']['gmres']['restart'] = 40

            prm[nsolver]['krylov_solver']['preconditioner'] = {}
            prm[nsolver]['krylov_solver']['preconditioner']['structure'] = 'same_nonzero_pattern'

            prm[nsolver]['krylov_solver']['preconditioner']['ilu'] = {}
            prm[nsolver]['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0

        return prm
           
        
    def get_displacement(self, name = "displacement", annotate = True):
        return self._compressibility.get_displacement(name, annotate)
    def get_u(self):
        if self._W.sub(0).num_sub_spaces() == 0:
            return self._w
        else:
            return split(self._w)[0]

    def get_state(self):
        return self._w

    def get_state_space(self):
        return self._W
    
    def reinit(self, w):
        self.get_state().assign(w, annotate=False)
        self._init_forms()
    

    def solve(self):
       
        # Get old state in case of non-convergence
        w_old = self.get_state().copy(True)
        try:
            # Try to solve the system
             solve(self._G == 0,
                   self._w,
                   self._bcs,
                   J = self._dG,
                   solver_parameters = self.parameters["solve"],
                   annotate = False)

        except RuntimeError:
            # Solver did not converge
            logger.warning("Solver did not converge")
            # Retrun the old state, and a flag crash = True
            self.reinit(w_old)
            return w_old, True

        else:
            # The solver converged
            
            # If we are annotating we need to annotate the solve as well
            if not parameters["adjoint"]["stop_annotating"]:

                self._w.assign(w_old)
                
                # Solve the system with annotation
                solve(self._G == 0,
                      self._w,
                      self._bcs,
                      J = self._dG,
                      solver_parameters = self.parameters["solve"],
                      annotate = True)

            # Return the new state, crash = False
            return self._w, False

    def internal_energy(self):
        return self._pi_int

    def P(self):
        """First Piola Stress Tensor
        """
        return diff(self._pi_int, self._F)

    def S(self):
        """Second Piola Stress Tensor
        """
        return inv(F)*self.P

    def Sigma(self):
        """Chaucy Stress Tensor
        """
        return (1.0/det(self._F))*self.P*inv(self._F.T)

    def _init_spaces(self):
        self._compressibility = get_compressibility(self.parameters)
            
        self._W = self._compressibility.W
        self._w = self._compressibility.w
        self._w_test = self._compressibility.w_test


    def _init_forms(self):

        material = self.parameters["material"]
        N =  self.parameters["facet_normal"]
        ds = Measure("exterior_facet", subdomain_data \
                     = self.parameters["facet_function"])

        # Displacement
        u = self._compressibility.get_displacement_variable()
        # Deformation gradient
        F = grad(u) + Identity(3)
        self._F = variable(F)
        J = det(self._F)

        # Isochoric Right Cauchy Green tensor
        Cbar = J**(-2.0/3.0)*self._F.T*self._F
        
        # Internal energy
        self._pi_int = material.strain_energy(F) + self._compressibility(J)      
        # Internal virtual work
        self._G = derivative(self._pi_int*dx, self._w, self._w_test)

        # External work
        v = self._compressibility.u_test

        # Neumann BC
        if self.parameters["bc"].has_key("neumann"):
            for neumann_bc in self.parameters["bc"]["neumann"]:
                p, marker = neumann_bc
                self._G += inner(J*p*dot(inv(F).T, N), v)*ds(marker)
        
        # Robin BC
        if self.parameters["bc"].has_key("robin"):
            for robin_bc in self.parameters["bc"]["robin"]:
                val, marker = robin_bc
                self._G += -inner(val*u, v)*ds(marker)
        
        # Other body forces
        if self.parameters.has_key("body_force"):
            self._G += -inner(self.parameters["body_force"], v)*dx

        # Dirichlet BC
        if self.parameters["bc"].has_key("dirichlet"):
            if hasattr(self.parameters["bc"]["dirichlet"], '__call__'):
                self._bcs = self.parameters["bc"]["dirichlet"](self._W)
            else:
                self._bcs = self._make_dirichlet_bcs()

        
        self._dG = derivative(self._G, self._w)

    def _make_dirichlet_bcs(self):
        bcs = []
        D = self._compressibility.get_displacement_space()
        for bc_spec in self.parameters["bc"]["dirichlet"]:
            val, marker = bc_spec
            if type(marker) == int:
                args = [D, val, self.parameters["facet_function"], marker]
            else:
                args = [D, val, marker]
            bcs.append(DirichletBC(*args))
        return bcs 



