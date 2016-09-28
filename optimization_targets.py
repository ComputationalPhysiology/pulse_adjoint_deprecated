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

__all__ = ["RegionalStrainTarget", "FullStrainTarget",
           "VolumeTarget"]


class OptimizationTarget(object):
    """Base class for optimization
    target
    """
        
    def assign_target(self, target, annotate=False):
        """Assing target value to target function

        :param target: new target
        """
        self.target_fun.assign(target, annotate=annotate)


class RegionalStrainTarget(OptimizationTarget):
    """Class for regional strain optimization
    target
    """
    def __init__(self, mesh, target_data, weights):
        """Initialize the functions

        :param mesh: The mesh
        :param target_data: The target data
        :param weights: Strain weights 
        """
        self.target_space = VectorFuntionSpace(mesh, "R", 0, dim = 17)
        self.realspace = FunctionSpace(mesh, "R", 0)
        self.weigths_arr = weigths

    def set_target_functions(self):
        self.target_fun = [Function(target_space,name = "Target Strains_{}".format(i)) \
                           for i in range(1,18)]
        self.diff = [Function(realspace, name = "Strain Difference_{}".format(i)) \
                     for i in range(1,18)]

        
        


        
    def assign_target(self, target, annotate=False):
        """Assing target regional strain

        :param target: Target regional strain
        """
        for fun, target in zip(self.target_fun, target):
            fun.assign(target, annotate = annotate)

    def get_form(self):
        lst= [(dot(self.weights[i],self.simulated[i] - self.target[i]))**2 \
              for i in range(17)]
        return list_sum(lst)
                                    
        

class FullStrainTarget(OptimizationTarget):
    """Class for full strain field
    optimization target
    """
    def get_form(self):
        return (self.target_fun - self.simulated_fun)**2
        
        

class GLStrainTarget(OptimizationTarget):
    """Class for global longitudinal
    strain optimization target
    """
    pass


class VolumeTarget(OptimizationTarget):
    """Class for volume optimization
    target
    """
    def __init__(self, mesh, traget_data):
        """Initialize the functions

        :param mesh: The mesh
        """
        self.realspace = FuntionSpace(mesh, "R", 0)
        self.target_space = self.realspace

    def set_target_functions(self)
        self.target_fun = Function(realspace, name = "Target Volume")
        self.diff = Function(realspace, name = "Volume Difference")


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
