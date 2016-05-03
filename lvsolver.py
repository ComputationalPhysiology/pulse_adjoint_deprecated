#!/usr/bin/env python
# Copyright (C) 2016 Henrik Finsberg
#
# This file is part of CAMPASS.
#
# CAMPASS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CAMPASS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CAMPASS. If not, see <http://www.gnu.org/licenses/>.

from dolfin import *
from dolfin_adjoint import *
from compressibility import get_compressibility
from adjoint_contraction_args import logger
from copy import deepcopy


class LVSolver(object):
    
    def __init__(self, params, use_snes = True, iterative_solver = False):        

        for k in ["mesh", "facet_function", "material", "bc"]:
            assert params.has_key(k), \
              "{} need to be in solver_parameters".format(k)

        
        self.parameters = params
        self.use_snes = use_snes
        self.iterative_solver = iterative_solver
        
        # Update solver parameters
        prm = self.default_solver_parameters()
        
        for k, v in params["solve"].iteritems():
            if isinstance(params["solve"][k], dict):
                for k_sub, v_sub in params["solve"][k].iteritems():
                    prm[k][k_sub]= v_sub

            else:
                prm[k] = v
            
        self.parameters["solve"] = prm
        
        
        self._init_spaces()
        self._init_forms()

    def default_solver_parameters(self):

        nsolver = "snes_solver" if self.use_snes else "newton_solver"

        prm = {"nonlinear_solver": "snes", "snes_solver":{}} if self.use_snes else {"nonlinear_solver": "newton", "newton_solver":{}}

        prm[nsolver]['absolute_tolerance'] = 1E-5
        prm[nsolver]['relative_tolerance'] = 1E-5
        prm[nsolver]['maximum_iterations'] = 8
        # prm[nsolver]['relaxation_parameter'] = 1.0
        prm[nsolver]['linear_solver'] = 'lu'
        prm[nsolver]['error_on_nonconvergence'] = True
        prm[nsolver]['report'] = True if logger.level < INFO else False
        if self.iterative_solver:
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

                # Assign the old state
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



