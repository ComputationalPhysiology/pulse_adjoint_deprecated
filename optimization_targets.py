"""
The functional you want to minimize consists of
different optimzation targets. 

It may consist of a volume-target and a regional strain-target
in which you functional may take the following form

functional = a*volume_target_form + b*strain_target_form

with

volume_target = VolumeTarget()
volume_target_form = volume_target.get_form()
"""

from dolfinimport import *
from utils import list_sum
import numpy as np
from numpy_mpi import *

__all__ = ["RegionalStrainTarget", "FullStrainTarget",
           "VolumeTarget"]


class OptimizationTarget(object):
    """Base class for optimization
    target
    """
    def __init__(self, mesh):

        self.realspace = FunctionSpace(mesh, "R", 0)
        
                                     
        self.meshvol = Constant(assemble(Constant(1.0)*dx(mesh)),
                                name = "mesh volume")
        
        self._trial = TrialFunction(self.target_space)
        self._test = TestFunction(self.target_space)

        self._trial_r = TrialFunction(self.realspace)
        self._test_r = TestFunction(self.realspace)
    
    def set_target_functions(self):
        """Initialize the functions
        """
        
        self.target_fun = Function(self.target_space,
                                   name = "Target {}".format(self._name))
        self.simulated_fun = Function(self.target_space,
                                      name = "Simulated {}".format(self._name))
        self.functional = Function(self.realspace,
                                   name = "{} Functional".format(self._name))
        self._set_form()
        
    def _set_form(self):
        self._form = (self.target_fun - self.simulated_fun)**2

    
    def assign_target(self, target, annotate=False):
        """Assing target value to target function

        :param target: new target
        """
        self.target_fun.assign(target, annotate=annotate)
        

    def assign_functional(self):
        solve(self._trial_r*self._test_r*dx == \
            self._test_r*self._form*dx, \
            self.functional)

    def get_functional(self):
        return (self.functional/self.meshvol)*dx


    def get_simulated(self):
        return self.simulated_fun

    def get_target(self):
        return self.target_fun

    def get_value(self):
        return gather_broadcast(self.functional.vector().array())[0]

    
        

class RegionalStrainTarget(OptimizationTarget):
    """Class for regional strain optimization
    target
    """                                  
    def __init__(self, mesh, crl_basis, dmu, weights=np.ones((17,3))):
        """Initialize regional strain target

        :param mesh: The mesh
        :param crl_basis: Basis function for the cicumferential, radial
                          and longituginal components
        :param dmu: Measure with subdomain information
        :param weights: Weights on the different segements
        
        """
        self._name = "Regional Strain"
        self.weight_space = TensorFunctionSpace(mesh, "R", 0)
        self.target_space = VectorFunctionSpace(mesh, "R", 0, dim = 3)
        self.weights_arr = weights
        self.crl_basis = crl_basis
        self.dmu = dmu
        OptimizationTarget.__init__(self, mesh)

    def set_target_functions(self):
        """Initialize the functions
        """
        
        self.target_fun = [Function(self.target_space,
                                    name = "Target Strains_{}".format(i)) \
                           for i in range(1,18)]

        self.simulated_fun = [Function(self.target_space,
                                       name = "Simulated Strains_{}".format(i)) \
                              for i in range(1,18)]

        self.functional = [Function(self.realspace,
                                    name = "Strains_{} Functional".format(i)) \
                           for i in range(1,18)]
        
        self.weights = [Function(self.weight_space, \
                                 name = "Strains Weights_{}".format(i)) \
                        for i in range(1,18)]
        strain_weight = np.zeros(9)
        strain_weight[0] = self.weights_arr[i-1][0]
        strain_weight[4] = self.weights_arr[i-1][1]
        strain_weight[8] = self.weights_arr[i-1][2]
        assign_to_vector(self.weights[i-1].vector(), strain_weight)
                
        self._set_form()

    def _set_form(self):
     
        self._form = [(dot(self.weights[i],self.simulated_fun[i] \
                           - self.target_fun[i]))**2 \
                     for i in range(17)]

    def get_value(self):
        return sum([gather_broadcast(self.functional[i].vector().array())[0] \
                    for i in range(17)])

        
    def assign_target(self, target, annotate=False):
        """Assing target regional strain

        :param target: Target regional strain
        """

        for fun, target in zip(self.target_fun, target):
            fun.assign(target, annotate = annotate)

    def assign_simulated(self, u):
        """Assing simulated regional strain

        :param u: New displacement
        """
        
        # Compute the strains
        gradu = grad(u)
        grad_u_diag = as_vector([inner(e,gradu*e) for e in self.crl_basis])

        # Make a project for dolfin-adjoint recording
        for i in range(17):
            solve(inner(self._trial, self._test)*self.dmu(i+1) == \
                  inner(grad_u_diag, self._test)*self.dmu(i+1), \
                  self.simulated_fun[i])

    def assign_functional(self):

        for i in range(17):
            solve(self._trial_r*self._test_r/self.meshvol*dx == \
                  self._test_r*self._form[i]*self.dmu(i+1), \
                  self.functional[i])


    def get_functional(self):
        return (list_sum(self.functional)/self.meshvol)*dx
                                    
        

class FullStrainTarget(OptimizationTarget):
    """Class for full strain field
    optimization target
    """
    def __init__(self):
        self._name = "Full Strain"
    
        

class GLStrainTarget(OptimizationTarget):
    """Class for global longitudinal
    strain optimization target
    """
    def __init__(self):
        self._name = "GL Strain"


class VolumeTarget(OptimizationTarget):
    """Class for volume optimization
    target
    """
    
    def __init__(self, mesh, dmu):
        """Initialize the functions

        :param mesh: The mesh
        :param mesh: Surface measure of the endocardium
        
        """
        self._name = "Volume"
        self.X = SpatialCoordinate(mesh)
        self.N = FacetNormal(mesh)
        self.dmu = dmu
        
        self.target_space = FunctionSpace(mesh, "R", 0)
        OptimizationTarget.__init__(self, mesh)

    def assign_simulated(self, u):
        """Assign simulated volume

        :param u: New displacement
        """

        # Compute volume
        F = grad(u) + Identity(3)
        J = det(F)
        vol = (-1.0/3.0)*dot(self.X + u, J*inv(F).T*self.N)

        # Make a project for dolfin-adjoint recording
        solve(inner(self._trial, self._test)*self.dmu == \
              inner(vol, self._test)*self.dmu, self.simulated_fun)
    
    def _set_form(self):
        self._form =  ((self.target_fun - self.simulated_fun)/self.target_fun)**2
        


class Regularization(object):
    """Class for regularization
    of the control parameter
    """
    pass

class RealValueProjector(object):
    """
    Projects onto a real valued function in order to force dolfin-adjoint to
    create a recording.
    """
    def __init__(self, u,v, mesh_vol):
        self.u_trial = u
        self.v_test = v
        self.mesh_vol = mesh_vol
    
        
    def project(self, expr, measure, real_function, mesh_vol_divide = True):

        if mesh_vol_divide:
            solve((self.u_trial*self.v_test/self.mesh_vol)*dx == \
                  self.v_test*expr*measure,real_function)
            
        else:
            solve((self.u_trial*self.v_test)*dx == \
              self.v_test*expr*measure,real_function)

        return real_function
