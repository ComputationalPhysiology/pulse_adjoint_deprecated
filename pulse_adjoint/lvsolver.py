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
from dolfinimport import *
from compressibility import get_compressibility
from adjoint_contraction_args import logger
from copy import deepcopy

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
            if params["solve"].has_key("nonlinear_solver") \
              and params["solve"]["nonlinear_solver"] == "newton":
                self.use_snes = False
            else:
                self.use_snes = True
              
            prm = self.default_solver_parameters()

            for k, v in params["solve"].iteritems():
                if isinstance(params["solve"][k], dict):
                    for k_sub, v_sub in params["solve"][k].iteritems():
                        prm[k][k_sub]= v_sub

                else:
                    prm[k] = v
        else:
            self.use_snes = False
            prm= self.default_solver_parameters()
            
        self.parameters["solve"] = prm
        
        
        self._init_spaces()
        self._init_forms()

        
    def postprocess(self):
        return Postprocess(self)
        
    def default_solver_parameters(self):

        nsolver = "snes_solver" if self.use_snes else "newton_solver"

        prm = {"nonlinear_solver": "snes", "snes_solver":{}} if self.use_snes else {"nonlinear_solver": "newton", "newton_solver":{}}

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
        return self.parameters["material"].gamma

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

                # Increase the tolerance slightly (don't know why we need to do this)
                nsolver = "snes_solver" if self.use_snes else "newton_solver"
                solver.parameters[nsolver]['relative_tolerance'] /= 0.001
                solver.parameters[nsolver]['absolute_tolerance'] /= 0.1
                # Solve the system with annotation
                try:
                    nliter, nlconv = solver.solve(annotate=True)
                except RuntimeError:
                    # Sometimes this throws a runtime error
                    solver.parameters[nsolver]['relative_tolerance'] *= 0.001
                    solver.parameters[nsolver]['absolute_tolerance'] *= 0.1
                    self.reinit(w_old, annotate=True)
                    raise  SolverDidNotConverge("Adjoint solve step didn't converge")


                else:
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
        ds = Measure("exterior_facet", subdomain_data \
                     = self.parameters["facet_function"])
        self._bcs = []

        dim = self.parameters["mesh"].topology().dim()
        
        # Displacement
        u = self._compressibility.get_displacement_variable()

        # Identity
        self._I = Identity(dim)
        
        # Deformation gradient
        F = grad(u) + self._I
        self._C = F.T * F
        self._E = 0.5*(self._C - self._I)

        
        self._F = variable(F)
        J = det(self._F)
        


        # # If model is compressible remove volumetric strains
        # if self.is_incompressible():
        F_iso = self._F
        # else:
        #     pass
        # F_iso = variable(pow(J, -float(1)/dim)*self._F)

                
        # Internal energy
        self._strain_energy = material.strain_energy(F_iso)
        self._pi_int = self._strain_energy + self._compressibility(J)


        # Testfunction for displacement
        du = self._compressibility.u_test
        dp = self._compressibility.p_test
        p = self._compressibility.p
                
        ## Internal virtual work
        self._G = derivative(self._pi_int*dx, self._w, self._w_test) 

        # This is the equivalent formulation
        # P = diff(self._strain_energy, F_iso)
        # self._G = inner(P, grad(du))*dx
        # self._G -= dp*(J-1)*dx
        # self._G -= p*J*inner(inv(F_iso).T, grad(du))*dx
        # self._G -= p*J*inner(inv(self._F).T, grad(du))*dx
        
        
        ## External work
        
        # Neumann BC
        if self.parameters["bc"].has_key("neumann"):
            for neumann_bc in self.parameters["bc"]["neumann"]:
                pressure, marker = neumann_bc
                self._G += inner(J*pressure*dot(inv(self._F).T, N), du)*ds(marker)


        # Other body forces
        if self.parameters["bc"].has_key("body_force"):
           
            self._G += -derivative(inner(self.parameters["bc"]["body_force"], u)*dx, u, v)
          
        
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


        # Weakly impose Dirichlet by Nitsches method
        # NOTE: THIS IS NOT TESTED
        if self.parameters["bc"].has_key("nitsche"):
            beta_value = 10
            beta = Constant(beta_value)
            
            h_E = CellSize(mesh)
            #MaxFacetEdgeLength(self.parameters["mesh"])
            for nitsche in self.parameters["bc"]["nitsche"]:
                
                val, dS = nitsche
                
                self._G += - inner(dot(grad(u), N), v)*dS \
                  + inner(u, dot(grad(v), N))*dS \
                  + beta*h_E**-1*inner(u, v)*dS \
                  - inner(val, dot(grad(v), N))*dS \
                  - beta*h_E**-1*inner(val, v)*dS
        
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

    def first_piola_stress(self):
        r"""
        First Piola Stress Tensor

        Incompressible:

        .. math::

           \mathbf{P} =  \frac{\partial \psi}{\partial \mathbf{F}} - pJ\mathbf{F}^{-T}

        Compressible:

        .. math::

           \mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}
        
        """
        return self.chaucy_stress()*inv(self._F.T)
        

    def second_piola_stress(self):
        r"""
        Second Piola Stress Tensor

        .. math::

           \mathbf{S} =  \mathbf{F}^{-1} \sigma \mathbf{F}^{-T}

        """
        return inv(self._F)*self.chaucy_stress()*inv(self._F.T)

    def chaucy_stress(self):
        r"""
        Chaucy Stress Tensor

        Incompressible:

        .. math::

           \sigma = \mathbf{F} \frac{\partial \psi}{\partial \mathbf{F}} - p\mathbf{I}

        Compressible:

        .. math::

           \sigma = \mathbf{F} \frac{\partial \psi}{\partial \mathbf{F}}
        
        """
        
        return self.solver.parameters["material"].CauchyStress(self._F, self._p)

    def deformation_gradient(self):
        return self._F
    
    def work(self):
        r"""
        Compute Work

        .. math::

           W = \mathbf{S} : \mathbf{E},

        with :math:`\mathbf{E}` being the Green-Lagrange strain tensor
        and :math:`\mathbf{E}` the second Piola stress tensor
        """
        
        return inner(self.GreenLagrange(), self.second_piola_stress())

    
        
    def work_fiber(self):
        r"""Compute Work in Fiber work

        .. math::

           W = \mathbf{S}_f : \mathbf{E}_f,
        """
        
        # Fibers
        f = self.solver.parameters["material"].f0

        # Fiber strain
        Ef = self.GreenLagrange()*f
        # Fiber stress
        Sf = self.second_piola_stress()*f

        return inner(Ef, Sf)


    def J(self):
        return det(self._F)
    
    def I1(self):
        """
        Return first isotropic invariant
        """
        return self.solver.parameters["material"].I1(self._F)

    def I4f(self):
        """
        Return the quasi-invariant in fiber direction
        """
        return self.solver.parameters["material"].I4f(self._F)

  
    def strain_energy(self):
        """
        Return the total strain energy
        """
        return self.solver.parameters["material"].strain_energy(self._F)
    
    def GreenLagrange(self):
        return self._E

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

        f0 = self.solver.parameters["material"].f0
        f =  self._F*f0
        
        return inner((self.chaucy_stress()*f)/f**2, f)
    
    def cauchy_stress_component(self, n0):

        # Push forward to current configuration
        n = self._F*n0
        return inner((self.chaucy_stress()*n)/n**2, n)

    def piola2_stress_component(self, n0):
        
        return inner((self.second_piola_stress()*n0)/n0**2, n0)

    def piola1_stress_component(self, n0):

        return inner((self.first_piola_stress()*n0)/n0**2, n0)


    def green_strain_component(self, n0):
        return inner(self.GreenLagrange()*n0/n0**2, n0)

    def deformation_gradient_component(self, n0):
        return inner(self._F*n0/n0**2, n0)

    def gradu_component(self, n0):
        return inner((self._F - self._I)*n0/n0**2, n0)


    def _localproject(self, fun, V) :
        a = inner(TestFunction(V), TrialFunction(V)) * dx
        L = inner(TestFunction(V), fun) * dx
        res = Function(V)
    
        solver = LocalSolver(a, L, to_annotate = False)
        solver.solve_local(res.vector(), assemble(L), V.dofmap())
        return res
        
