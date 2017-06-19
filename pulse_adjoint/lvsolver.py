#!/usr/bin/env python
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
from copy import deepcopy
from .dolfinimport import *
from .adjoint_contraction_args import logger
from .kinematics import *
from models.compressibility import get_compressibility

class SolverDidNotConverge(Exception):
    pass


class LVSolver(object):
    """
    A Cardiac Mechanics Solver
    """
    
    def __init__(self, params):

        for k in ["mesh", "facet_function", "material", "bc"]:
            assert params.has_key(k), \
              "{} need to be in solver_parameters".format(k)

        
        self.parameters = params

        # Krylov solvers does not work
        self.iterative_solver = False

        # Update solver parameters
        if params.has_key("solve"):              
            prm = self.default_solver_parameters()

            for k, v in params["solve"].iteritems():
                if isinstance(params["solve"][k], dict):
                    for k_sub, v_sub in params["solve"][k].iteritems():
                        prm[k][k_sub]= v_sub

                else:
                    prm[k] = v
        else:
            prm= self.default_solver_parameters()
            
        self.parameters["solve"] = prm

        self.relax_adjoint_solver = True if not params.has_key("relax_adjoint_solver") \
                                    else params["relax_adjoint_solver"]
        
        self._init_spaces()
        self._init_forms()

        
    def postprocess(self):
        return Postprocess(self)
        
    def default_solver_parameters(self):

        nsolver = "newton_solver"

        prm = {"nonlinear_solver": "newton", "newton_solver":{}}

        prm[nsolver]['absolute_tolerance'] = 1E-8
        prm[nsolver]['relative_tolerance'] = 1E-12
        prm[nsolver]['maximum_iterations'] = 15
        # prm[nsolver]['relaxation_parameter'] = 1.0
        prm[nsolver]['linear_solver'] = 'mumps'
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

    def get_hydrostatic_pressue(self, name = "hydrostatic_pressure", annotate = True):
        return self._compressibility.get_hydrostatic_pressue(name, annotate)
    
    def get_u(self):
        if self._W.sub(0).num_sub_spaces() == 0:
            return self._w
        else:
            return split(self._w)[0]

    def get_gamma(self):
        return self.parameters["material"].get_gamma()

    def is_incompressible(self):
        return self._compressibility.is_incompressible()

    def get_state(self):
        return self._w

    def get_state_space(self):
        return self._W
    
    def reinit(self, w, annotate=False):
        """
        *Arguments*
          w (:py:class:`dolfin.GenericFunction`)
            The state you want to assign

        Assign given state, and reinitialize variaional form.
        """
        self.get_state().assign(w, annotate=annotate)
        self._init_forms()
    

    def solve(self):
        r"""
        Solve the variational problem

        .. math::

           \delta W = 0

        """
        # Get old state in case of non-convergence
        w_old = self.get_state().copy(True)
        problem = NonlinearVariationalProblem(self._G, self._w,
                                              self._bcs,
                                              self._dG)


        solver = NonlinearVariationalSolver(problem)
        solver.parameters.update(self.parameters["solve"])
        
        try:
            
            nliter, nlconv = solver.solve(annotate=False)
            if not nlconv:
                raise RuntimeError("Solver did not converge...")

        except RuntimeError as ex:
            logger.debug(ex)
            
            # Solver did not converge
            logger.warning("Solver did not converge")
            # Reinitialze forms with old state
            self.reinit(w_old)
            # Raise my own exception, so that other
            # in order to separate this exepction from
            # other RuntimeErrors
            raise SolverDidNotConverge(ex)

        else:
           
            # The solver converged
            # If we are annotating we need to annotate the solve as well
            if not parameters["adjoint"]["stop_annotating"]:

                if self.relax_adjoint_solver:
                    # Increase the tolerance slightly
                    # (don't know why we need to do this)
                    nsolver =  "newton_solver"
                    solver.parameters[nsolver]['relative_tolerance'] /= 0.001
                    solver.parameters[nsolver]['absolute_tolerance'] /= 0.1
                    
                # Solve the system with annotation
                try:
                    nliter, nlconv = solver.solve(annotate=True)
                except RuntimeError:
                    # Sometimes this throws a runtime error
                    if self.relax_adjoint_solver:
                        solver.parameters[nsolver]['relative_tolerance'] *= 0.001
                        solver.parameters[nsolver]['absolute_tolerance'] *= 0.1
                    self.reinit(w_old, annotate=True)
                    raise  SolverDidNotConverge("Adjoint solve step didn't converge")


                else:
                    if self.relax_adjoint_solver:
                        solver.parameters[nsolver]['relative_tolerance'] *= 0.001
                        solver.parameters[nsolver]['absolute_tolerance'] *= 0.1
                    if not nlconv:
                        raise  SolverDidNotConverge("Adjoint solve step didn't converge")

                
            return nliter, nlconv

    
        
    
    def _init_spaces(self):
        """
        Initialize function spaces
        """
        
        self._compressibility = get_compressibility(self.parameters)
            
        self._W = self._compressibility.W
        self._w = self._compressibility.w
        self._w_test = self._compressibility.w_test


    def _init_forms(self):
        r"""
        Initialize variational form

        """
        material = self.parameters["material"]
        N =  self.parameters["facet_normal"]
        X = SpatialCoordinate(self.parameters["mesh"])
        ds = Measure("exterior_facet", subdomain_data \
                     = self.parameters["facet_function"])
        self._bcs = []

        dim = self.parameters["mesh"].topology().dim()
        
        # Displacement
        u = self._compressibility.get_displacement_variable()

        # Identity
        self._I = Identity(dim)
        
        # Deformation gradient
        self._F = variable(grad(u) + self._I)
        self._C = self._F.T * self._F
        self._E = 0.5*(self._C - self._I)
        J = det(self._F)
                
        # Internal energy
        self._strain_energy = material.strain_energy(self._F)
        self._pi_int = self._strain_energy + self._compressibility(J)
                       


        # Testfunction for displacement
        du = self._compressibility.u_test
        dp = self._compressibility.p_test
        p = self._compressibility.p
                
        ## Internal virtual work
        self._G = derivative(self._pi_int*dx, self._w, self._w_test) 

        
        ## External work
        
        # Neumann BC
        
        if self.parameters["bc"].has_key("neumann"):
            for neumann_bc in self.parameters["bc"]["neumann"]:
                pressure, marker = neumann_bc
                n = pressure*cofac(self._F) * N
                
                self._G += inner(du, n)*ds(marker)
         
                
        # Other body forces
        if self.parameters["bc"].has_key("body_force"):           
            self._G += -derivative(inner(self.parameters["bc"]["body_force"], u)*dx, u, du)

        
        # Robin BC
        if self.parameters["bc"].has_key("robin"):
            for robin_bc in self.parameters["bc"]["robin"]:
                if robin_bc is not None:
                    val, marker = robin_bc
                    self._G += inner(val*u, du)*ds(marker)
        
       
        # Penalty term
        if self.parameters["bc"].has_key("penalty"):
            if hasattr(self.parameters["bc"]["penalty"], '__call__'):
                
                penalty = self.parameters["bc"]["penalty"](u)
                self._G += derivative(penalty, self._w, self._w_test)

        # Dirichlet BC
        if self.parameters["bc"].has_key("dirichlet"):
            if hasattr(self.parameters["bc"]["dirichlet"], '__call__'):
                self._bcs = self.parameters["bc"]["dirichlet"](self._W)
            else:
                self._bcs = self._make_dirichlet_bcs()

        self._dG = derivative(self._G, self._w, TrialFunction(self._W))

    def _make_dirichlet_bcs(self):
        bcs = []
        D = self._compressibility.get_displacement_space()
        for bc_spec in self.parameters["bc"]["dirichlet"]:
            if isinstance(bc_spec, DirichletBC):
                bcs.append(bc_spec)
            else:
                val, marker = bc_spec
                if type(marker) == int:
                    args = [D, val, self.parameters["facet_function"], marker]
                else:
                    args = [D, val, marker]
                bcs.append(DirichletBC(*args))
                
        return bcs



class Postprocess(object):
    def __init__(self, solver):
        self.solver = solver
        
        self._F = self.solver._F
        self._C = self.solver._C
        self._E = self.solver._E
        self._I = self.solver._I
        self._p = self.solver._compressibility.p


    def internal_energy(self):
        """
        Return the total internal energy
        """
        return self.solver._pi_int

    def first_piola_stress(self, deviatoric=False):
        r"""
        First Piola Stress Tensor

        Incompressible:

        .. math::

           \mathbf{P} =  \frac{\partial \psi}{\partial \mathbf{F}} - pJ\mathbf{F}^{-T}

        Compressible:

        .. math::

           \mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}
        
        """
        return self.solver.parameters["material"].FirstPiolaStress(self._F, self._p, deviatoric)
        

    def second_piola_stress(self):
        r"""
        Second Piola Stress Tensor

        .. math::

           \mathbf{S} =  \mathbf{F}^{-1} \sigma \mathbf{F}^{-T}

        """
        return inv(self._F) * self.first_piola_stress()


    def chaucy_stress(self, deviatoric=False):
        r"""
        Chaucy Stress Tensor

        Incompressible:

        .. math::

           \sigma = \mathbf{F} \frac{\partial \psi}{\partial \mathbf{F}} - p\mathbf{I}

        Compressible:

        .. math::

           \sigma = \mathbf{F} \frac{\partial \psi}{\partial \mathbf{F}}
        
        """
        
        return self.solver.parameters["material"].CauchyStress(self._F, self._p, deviatoric)

    def deformation_gradient(self):
        return self._F
    

    def J(self):
        return det(self._F)

  
    def strain_energy(self):
        """
        Return the total strain energy
        """
        return self.solver.parameters["material"].strain_energy(self._F)
    
    def GreenLagrange(self, F_ref = None):
        
        if F_ref is None:
            F = self._F
        else:
            F = self._F*inv(F_ref)
            
        C = F.T*F
        E = 0.5 * (C - self._I)
        return E

    def fiber_strain(self):
        r"""Compute Fiber strain

        .. math::

           \mathbf{E}_{f} = f \cdot \mathbf{E} f,

        with :math:`\mathbf{E}` being the Green-Lagrange strain tensor
        and :math:`f` the fiber field on the current configuration

        """
        f =  self.solver.parameters["material"].f0
        return inner(self.GreenLagrange()*f/f**2, f)
        

    def fiber_stress(self):
        r"""Compute Fiber stress

        .. math::

           \sigma_{f} = f \cdot \sigma f,

        with :math:`\sigma` being the Chauchy stress tensor
        and :math:`f` the fiber field on the current configuration

        """

        f0 = self.solver.parameters["material"].get_component("fiber")
        return self.cauchy_stress_component(f0)
    
    def cauchy_stress_component(self, n0, deviatoric=False):

        # Push forward to current configuration
        n = self._F*n0
        return inner((self.chaucy_stress(deviatoric)*n)/n**2, n)

    def piola2_stress_component(self, n0):
        
        return inner((self.second_piola_stress()*n0)/n0**2, n0)

    def piola1_stress_component(self, n0):

        return inner((self.first_piola_stress()*n0)/n0**2, n0)


    def green_strain_component(self, n0, F_ref=None):
        return inner(self.GreenLagrange(F_ref)*n0/n0**2, n0)

    def deformation_gradient_component(self, n0):
        return inner(self._F*n0/n0**2, n0)

    def gradu_component(self, n0):
        return inner((self._F - self._I)*n0/n0**2, n0)




