"""
The functional you want to minimize consists of
different optimzation targets. 

It may consist of a volume-target and a regional strain-target
in which you functional may take the following form::

    functional = a*volume_target_form + b*strain_target_form

with::

    volume_target = VolumeTarget()
    volume_target_form = volume_target.get_form()

"""

from dolfinimport import *
from utils import list_sum
import numpy as np
from numpy_mpi import *


__all__ = ["RegionalStrainTarget", "FullStrainTarget",
           "VolumeTarget", "Regularization"]

class OptimizationTarget(object):
    """Base class for optimization target
    """
    def __init__(self, mesh):
        """
        Initialize base class for optimization targets

        :param mesh: The underlying mesh
        :type mesh: :py:class:`dolfin.Mesh`

        """

        # A real space for projecting the functional
        self.realspace = FunctionSpace(mesh, "R", 0)
        
        # The volume of the mesh
        self.meshvol = Constant(assemble(Constant(1.0)*dx(mesh)),
                                name = "mesh volume")

        # Test and trial functions for the target space
        self._trial = TrialFunction(self.target_space)
        self._test = TestFunction(self.target_space)

        # Test and trial functions for the real space
        self._trial_r = TrialFunction(self.realspace)
        self._test_r = TestFunction(self.realspace)

        # List for the target data
        self.data = []

        # List for saved data
        self.results = {"func_value": [],
                        "target": [],
                        "simulated":[]}
        self.reset()

    def reset(self):
        self.func_value = 0.0
        self.results["target"] = []
        self.results["simulated"] = []

    def save(self):
        self.func_value += self.get_value()
        self.results["func_value"].append(self.func_value)
        self.results["target"].append(Vector(self.target_fun.vector()))
        self.results["simulated"].append(Vector(self.simulated_fun.vector()))
        
    def next_target(self, it, annotate=False):
        self.assign_target(self.data[it], annotate)
        
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

    def load_target_data(self, target_data, n):
        """Load the target data

        :param target_data: The data
        :param n: Index

        """
        f = Function(self.target_space)
        assign_to_vector(f.vector(), np.array(target_data[n]))
        self.data.append(f)
        
    def _set_form(self):
        """The default form is just the least square
        difference
        """
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
        """Return the integral form of the functional
        We devide by the volume, so that when integrated
        the value of the functional is the value of the
        integral.
        """
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
    def __init__(self, mesh, crl_basis, dmu, weights=None, nregions = None,
                 tensor="gradu", F_ref = None):
        """
        Initialize regional strain target

        Parameters
        ----------
        mesh: :py:class:`dolfin.Mesh`
            The mesh
        crl_basis: dict
            Basis function for the cicumferential, radial
            and longituginal components
        dmu: :py:class:`dolfin.Measure`
            Measure with subdomain information
        weights: :py:function:`numpy.array`
            Weights on the different segements
        nregions: int
            Number of strain regions
        tensor: str
            Which strain tensor to use, e.g gradu, E, C, F
        F_ref: :py:class:`dolfin.Function`
            Tensor to map strains to reference
        
        """
        self._name = "Regional Strain"
        self._tensor = tensor
        

        if weights is None: weights = np.ones((17,3))
        self.nregions = np.shape(weights)[0] if nregions is None else nregions
        dim = mesh.geometry().dim()
        self.dim = dim
        self._F_ref = F_ref if F_ref is not None else Identity(dim)
        
        self.target_space = VectorFunctionSpace(mesh, "R", 0, dim = dim)
        
        
        self.weight_space = TensorFunctionSpace(mesh, "R", 0)
       
        if weights.shape == (self.nregions, dim):
            self.weights_arr = weights
        else:
            from adjoint_contraction_args import logger
            msg = "Weights do not correspond to the number of regions and dimension.\n"+\
                  "Dim = {}, number of regions = {}, {} and {} was given".format(dim,
                                                                                 self.nregions,
                                                                                 weights.shape[1],
                                                                                 weights.shape[0])
            logger.debug(msg)
            self.weights_arr =np.ones((self.nregions,dim))

        self.crl_basis = []
        for l in ["circumferential", "radial", "longitudinal"]:
            if crl_basis.has_key(l):
                self.crl_basis.append(crl_basis[l])
        
        self.dmu = dmu

        self.meshvols = [Constant(assemble(Constant(1.0)*dmu(i+1)),
                                  name = "mesh volume") for i in range(self.nregions)]
        
        OptimizationTarget.__init__(self, mesh)

    def print_head(self):
        return "\t{:<10}".format("I_strain")
    def print_line(self):
        I = self.get_value()
        return "\t{:<10.2e}".format(I)

    def save(self):
        
        self.func_value += self.get_value()
        self.results["func_value"].append(self.func_value)
        target = []
        simulated = []
        for i in range(self.nregions):
            
            target.append(Vector(self.target_fun[i].vector()))
            simulated.append(Vector(self.simulated_fun[i].vector()))

        
        
        self.results["target"].append(target)
        self.results["simulated"].append(simulated)

    def load_target_data(self, target_data, n):
        """Load the target data

        :param dict target_data: The data
        :param int n: Index

        """
        strains = []
        for i in range(self.nregions):
            f = Function(self.target_space)
            if target_data.has_key(i+1):
                assign_to_vector(f.vector(), np.array(target_data[i+1][n]))
                strains.append(f)
            
        self.data.append(strains)

    def set_target_functions(self):
        """
        Initialize the functions

        """
        
        self.target_fun = [Function(self.target_space,
                                    name = "Target Strains_{}".format(i+1)) \
                           for i in range(self.nregions)]

        self.simulated_fun = [Function(self.target_space,
                                       name = "Simulated Strains_{}".format(i+1)) \
                              for i in range(self.nregions)]

        self.functional = [Function(self.realspace,
                                    name = "Strains_{} Functional".format(i+1)) \
                           for i in range(self.nregions)]
        
        self.weights = [Function(self.weight_space, \
                                 name = "Strains Weights_{}".format(i+1)) \
                        for i in range(self.nregions)]


        self._set_weights()        
        self._set_form()

    def _set_weights(self):

        for i in range(self.nregions):
            weight = np.zeros(self.dim**2)
            weight[0::(self.dim+1)] = self.weights_arr[i]
            assign_to_vector(self.weights[i].vector(), weight)



    def _set_form(self):

    
        self._form = [(dot(self.weights[i],self.simulated_fun[i] \
                           - self.target_fun[i]))**2 \
                     for i in range(self.nregions)]

    def get_value(self):
        return sum([gather_broadcast(self.functional[i].vector().array())[0] \
                    for i in range(self.nregions)])

        
    def assign_target(self, target, annotate=False):
        """Assing target regional strain

        :param target: Target regional strain
        :type target: list of :py:class:`dolfin.Function`
        """

        for fun, target in zip(self.target_fun, target):
            fun.assign(target, annotate = annotate)

    def assign_simulated(self, u):
        """Assing simulated regional strain

        :param u: New displacement
        :type u: :py:class:`dolfin.Function`
        """
        
        I = Identity(self.dim)
        F = (grad(u) + I)*inv(self._F_ref)
        # Compute the strains
        if self._tensor == "gradu":
            tensor = F - Identity(self.dim)
        elif self._tensor == "E":
            C = F.T * F
            tensor = 0.5*(C-I)


        tensor_diag = as_vector([inner(e,tensor*e) for e in self.crl_basis])

        # Make a project for dolfin-adjoint recording
        for i in range(self.nregions):
            solve(inner(self._trial, self._test)*self.dmu(i+1) == \
                  inner(tensor_diag, self._test)*self.dmu(i+1), \
                  self.simulated_fun[i], solver_parameters={"linear_solver": "gmres"})
            
    def assign_functional(self):
        
        for i in range(self.nregions):
            solve(self._trial_r*self._test_r/self.meshvol*dx == \
                  self._test_r*self._form[i]/self.meshvols[i]*self.dmu(i+1), \
                  self.functional[i])

       
    def get_functional(self):
        return (list_sum(self.functional)/self.meshvol)*dx

                                          
class DisplacementTarget(OptimizationTarget):
    def __init__(self, mesh):
        self._name = "Displacement"
        self.dmu = dx(mesh)
        self.target_space = VectorFunctionSpace(mesh, "CG", 2)
        OptimizationTarget.__init__(self, mesh)

    def assign_simulated(self, u):
        """Assing simulated regional strain

        :param u: New displacement
        """

        # Make a project for dolfin-adjoint recording
        solve(inner(self._trial, self._test)*self.dmu == \
              inner(u, self._test)*self.dmu, \
              self.simulated_fun)

        
    
class FullStrainTarget(OptimizationTarget):
    """Class for full strain field
    optimization target
    """
    def __init__(self, mesh, crl_basis):
        self._name = "Full Strain"
        self.dmu = dx(mesh)
        self.crl_basis = crl_basis
        self.target_space = VectorFunctionSpace(mesh, "CG", 1, dim = 3)
        OptimizationTarget.__init__(self, mesh)

    def assign_simulated(self, u):
        """Assing simulated strain

        :param u: New displacement
        """
        
        # Compute the strains
        gradu = grad(u)
        grad_u_diag = as_vector([inner(e,gradu*e) for e in self.crl_basis])

        # Make a project for dolfin-adjoint recording
        solve(inner(self._trial, self._test)*self.dmu == \
              inner(grad_u_diag, self._test)*self.dmu, \
              self.simulated_fun)
    

class VolumeTarget(OptimizationTarget):
    """Class for volume optimization
    target
    """
    
    def __init__(self, mesh, dmu, chamber = "LV"):
        """Initialize the functions

        :param mesh: The mesh
        :param mesh: Surface measure of the endocardium
        
        """
        self._name = "{} Volume".format(chamber)
        self._X = SpatialCoordinate(mesh)
        self._N = FacetNormal(mesh)
        
        self._interpolation_space = VectorFunctionSpace(mesh,"CG", 1)
        self._disp_space = VectorFunctionSpace(mesh,"CG", 2)

        self.dmu = dmu
        self.chamber = chamber
        
        self.target_space = FunctionSpace(mesh, "R", 0)
        self.endoarea = Constant(assemble(Constant(1.0)*dmu),
                                 name = "endo area")
        OptimizationTarget.__init__(self, mesh)

    def print_head(self):
        return "\t{:<18}\t{:<20}\t{:<10}".format("Target {} Volume".format(self.chamber),
                                                 "Simulated {} Volume".format(self.chamber),
                                                 "I_{}".format(self.chamber))
    def print_line(self):
        v_sim = gather_broadcast(self.simulated_fun.vector().array())[0]
        v_meas = gather_broadcast(self.target_fun.vector().array())[0]
        I = self.get_value()
        
        return "\t{:<18.2f}\t{:<20.2f}\t{:<10.2e}".format(v_meas, v_sim, I)
        
    def load_target_data(self, target_data, n):
        """Load the target data

        :param target_data: The data
        :param n: Index

        """
        f = Function(self.target_space)
        assign_to_vector(f.vector(), np.array([target_data[n]]))
        self.data.append(f)

    def assign_simulated(self, u):
        """Assign simulated volume

        :param u: New displacement
        """
        # u_int = interpolate(project(u, self._disp_space),
                            # self._interpolation_space)
        u_int =project(u, self._interpolation_space)
        
        # Compute volume
        F = grad(u_int) + Identity(3)
        J = det(F)
        vol = (-1.0/3.0)*dot(self._X + u_int, J*inv(F).T*self._N)

        # Make a project for dolfin-adjoint recording
        solve(inner(self._trial, self._test)/self.endoarea*self.dmu == \
              inner(vol, self._test)*self.dmu, self.simulated_fun)
    
    def _set_form(self):
        self._form =  ((self.target_fun - self.simulated_fun)/self.target_fun)**2
        
class GLStrainTarget(OptimizationTarget):
    """Class for global longitudinal
    strain optimization target
    """
    def __init__(self):
        self._name = "GL Strain"


class Regularization(object):
    """Class for regularization
    of the control parameter
    """
    def __init__(self, mesh, spacestr = "CG_1", lmbda = 0.0, regtype = "L2_grad", mshfun = None):
        """Initialize regularization object

        :param space: The mesh
        :param space: Space for the regularization
        :param lmbda: regularization parameter
        
        """
        # assert spacestr in ["CG_1", "regional", "R_0"], \
        #     "Unknown regularization space {}".format(space)
        
        self.spacestr = spacestr
        self.lmbda = lmbda
        self._value = 0.0

    
        self._mshfun = mshfun if mshfun is not None \
                       else MeshFunction("size_t",mesh,  mesh.geometry().dim(), mesh.domains()) 

        self.meshvol = Constant(assemble(Constant(1.0)*dx(mesh)),
                                name = "mesh volume")
        self._regtype = regtype
        # A real space for projecting the functional
        self._realspace = FunctionSpace(mesh, "R", 0)

        if spacestr == "regional":
            self._space = FunctionSpace(mesh, "DG", 0)
        else:
            family, degree = spacestr.split("_")
            self._space = FunctionSpace(mesh, family, int(degree))
        
            
        
        
        self.dx = dx(mesh)
        self.results = {"func_value":[]}
        self.reset()

    def print_head(self):
        return "\t{:<10}".format("I_reg")
    def print_line(self):
        I = self.get_value()
        return "\t{:<10.2e}".format(I)

    def reset(self):
       
        self.func_value = 0.0
        self._value = 0.0

    def save(self):
        
        self.func_value += self.get_value()
        self.results["func_value"].append(self.func_value)

    def set_target_functions(self):
        
        self.functional = Function(self._realspace, name = "regularization_functional")
        if self.spacestr == "regional":
            from setup_optimization import RegionalParameter
            self._m = RegionalParameter(self._mshfun)
        else:
            self._m = Function(self._space)

    def get_form(self):
        """Get the ufl form

        :returns: The functional form
        :rtype: (:py:class:`ufl.Form`)

        """

        if self._regtype == "L2":
            
            return (inner(self._m, self._m)/self.meshvol)*self.dx

        else:
            if self.spacestr in ["CG_1", "regional"]:
                
                return (inner(grad(self._m), grad(self._m))/self.meshvol)*self.dx
            
            else:
                return Constant(0.0)*self.dx

            
    def assign(self, m, annotate = False):
        self._m.assign(m, annotate = annotate)
        
    def get_functional(self):
        """Get the functional form 
        (included regularization parameter)

        :param m: The function to be regularized
        :returns: The functional form
        :rtype: (:py:class:`ufl.Form`)

        """
        # raise Exception
        try:
            form = self.get_form()
            self._value = assemble(form)
        except:
            from IPython import embed; embed()
            exit()
        return self.lmbda*form

        
    def get_value(self):
        """Get the value of the regularization term
        without regularization parameter

        :param m: The function to be regularized
        :returns: The value of the regularization term
        :rtype: float

        """
        return self._value
        


# class RegionalStrainComponentTarget(RegionalStrainTarget):
#     """Class for one component regional strain optimization
#     target
#     """                                  
#     def __init__(self, mesh, crl_basis, dmu, weights=np.ones((17,3)), nregions = None):
#         """Initialize regional strain target

#         :param mesh: The mesh
#         :type mesh: :py:class:`dolfin.Mesh`
#         :param comp: A vectorfield with the strain component
#         :type comp: :py:class:`dolfin.Function`
#         :type mesh: :py:class:`dolfin.Mesh`
#         :param dmu: Measure with subdomain information
#         :type dmu: :py:class:`dolfin.Measure`
#         :param weights: Weights on the different segements
#         :type wieghts: :py:function:`numpy.array`
        
#         """
#         self._name = "Regional Strain"
#         self.nregions = np.shape(weights)[0] if nregions is None else nregions
#         dim = mesh.geometry().dim()
#         self.dim = dim
        
#         self.target_space = FunctionSpace(mesh, "R", 0)
#         self.weight_space = FunctionSpace(mesh, "R", 0)
       
#         if weights.shape == (self.nregions, 1):
#             self.weights_arr = weights
#         else:
#             from adjoint_contraction_args import logger
#             msg = "Weights do not correspond to the number of regions and dimension.\n"+\
#                   "Dim = {}, number of regions = {}, {} and {} was given".format(dim,
#                                                                                  self.nregions,
#                                                                                  weights.shape[1],
#



#                                                                                  weights.shape[0])
#             logger.warning(msg)
#             self.weights_arr =np.ones((self.nregions,1))

#         self.comp = comp        
#         self.dmu = dmu

#         self.meshvols = [Constant(assemble(Constant(1.0)*dmu(i+1)),
#                                   name = "mesh volume") for i in range(self.nregions)]
        
#         OptimizationTarget.__init__(self, mesh)

#     def _set_weights():
#         pass
#         # for i in range(self.nregions):
#         #     weights

#     def assign_simulated(self, u):
#         """Assing simulated regional strain

#         :param u: New displacement
#         :type u: :py:class:`dolfin.Function`
#         """
        
#         # Compute the strains
#         gradu = grad(u)
#         grad_u_diag = as_vector(inner(e,gradu*e) for e in self.crl_basis])

#         # Make a project for dolfin-adjoint recording
#         for i in range(self.nregions):
#             solve(inner(self._trial, self._test)*self.dmu(i+1) == \
#                   inner(grad_u_diag, self._test)*self.dmu(i+1), \
#                   self.simulated_fun[i])

#     def assign_functional(self):
        
#         for i in range(self.nregions):
#             solve(self._trial_r*self._test_r/self.meshvol*dx == \
#                   self._test_r*self._form[i]/self.meshvols[i]*self.dmu(i+1), \
#                   self.functional[i])

       
#     def get_functional(self):
#         return (list_sum(self.functional)/self.meshvol)*dx
