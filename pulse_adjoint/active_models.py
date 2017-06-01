import dolfin as df
import dolfin_adjoint as da

from utils import get_dimesion
from kinematics import *


class ActiveModel(Invariants):
    def __init__(self, gamma=None,
                 f0 = None, s0 = None,
                 n0 = None, T_ref=None,
                 isochoric=True, *args):

        # Fiber system
        self._f0 = f0
        self._s0 = s0
        self._n0 = n0

        
        self._gamma = da.Constant(0, name="gamma") if gamma \
                      is None else gamma
        
        self._T_ref =  df.Constant(T_ref) if T_ref\
                       else df.Constant(1.0)

        Invariants.__init__(self, isochoric, *args)

    def get_model_type(self):
        return self._model

    def get_activation(self):

        from setup_optimization import RegionalParameter
        # Activation
        if isinstance(self._gamma, RegionalParameter):
            # This means a regional gamma
            # Could probably make this a bit more clean
            gamma = self._gamma.get_function()
        else:
            gamma = self._gamma

        return self._T_ref*gamma


    def get_gamma(self):
        return self._gamma

    def get_component(self, component):
        
        assert component in ["fiber", "sheet", "sheet_normal"]
        if component == "fiber":
            return self._f0
        elif component == "sheet":
            return self._s0
        else:
            return self._n0

    

class ActiveStress(ActiveModel):
    """
    Active stress model
    """
    _model = "active_stress"
    def __init__(self, *args, **kwargs):

        
        self._axial_stress = kwargs.pop("axial_stress", "uniaxial")
        assert self._axial_stress in ["uniaxial", "biaxial"]
        ActiveModel.__init__(self, *args, **kwargs)
        
    def Wactive(self, F, diff = 0):

        C = F.T*F
        f0 = self.get_component("fiber")
        I4f = inner(C*f0, f0)
        gamma = self.get_activation()
        
        if diff == 0:
            return 0.5*gamma*(I4f-1)

        elif diff == 1:
            return gamma

    def type(self):
        return "ActiveStress"

    def I1(self, F, *args):
        return self._I1(F)
    
    def I4(self, F, component = "fiber", *args):
        
        a0 = self.get_component(component)
        return self._I4(F, a0)
    
    def Fa(self):
        return SecondOrderIdentity(self._f0)
    
    def Fe(self, F):
        return F
    
    

class ActiveStrain(ActiveModel):
    """
    This class implements the elastic invariants within
    the active strain framework

    Assuming transversally isotropic material for now

    """
    _model = "active_strain"
    

    def _mgamma(self):
        gamma = self.get_activation()

        # FIXME: should allow for different active strain models
        if 1:
            mgamma = 1 - gamma
        elif self._model == "rossi":
            mgamma = 1 + gamma

        return mgamma

    def Wactive(self, *args, **kwargs):
        return 0

    def I1(self, F):

        I1 = self._I1(F)
        f0 = self.get_component("fiber")
        I4f = self._I4(F, f0)

        d = get_dimesion(F)
        mgamma = self._mgamma()

        I1e = pow(mgamma, 4-d) * I1 +\
              (1/mgamma**2 - pow(mgamma, 4-d)) * I4f
        
        return  I1e


    def I4(self, F, component = "fiber"):
        r"""
        Quasi-invariant in the elastic configuration
        Let :math:`d` be the geometric dimension.
        If

        .. math:: 

           \mathbf{F}_a = (1 - \gamma) \mathbf{f}_0 \otimes \mathbf{f}_0  + 
           \frac{1}{\sqrt{1 - \gamma}} (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0)

        then

        .. math::

           I_{4f_0}^E = I_{4f_0} \frac{1}{(1+\gamma)^2}

        If 

        .. math:: 

           \mathbf{F}_a = (1 + \gamma) \mathbf{f}_0 \otimes \mathbf{f}_0  + 
           \frac{1}{\sqrt{1 + \gamma}} (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0)

        then

        .. math::

           I_{4f_0}^E = I_{4f_0} \frac{1}{(1+\gamma)^2}


        """

        a0  = self.get_component(component)
        I4f = self._I4(F, a0)
        mgamma = self._mgamma()

        I4a0 = 1/mgamma**2 * I4f
    
        return I4a0

    def Fa(self):

        
        f0  = self.get_component("fiber")
        d = get_dimesion(f0)
        f0f0 = df.outer(f0,f0)
        I = Identity(3)

        mgamma = self._mgamma()
        Fa = mgamma*f0f0 + pow(mgamma, -1.0/float(d-1)) * (I - f0f0)
        
        return Fa
    
    def Fe(self, F):

        Fa = self.Fa()
        Fe = F*df.inv(Fa)

        return Fe

    
        
if __name__ == "__main__":

    from patient_data import LVTestPatient
    patient = LVTestPatient()

    

    from setup_parameters import setup_general_parameters
    setup_general_parameters()
    

    V = df.VectorFunctionSpace(patient.mesh, "CG", 2)
    u0 = df.Function(V)
    # u1 = df.Function(V, "../tests/data/inflate_mesh_simple_1.xml")

    I = df.Identity(3)
    F0 = df.grad(u0) + I
    # F1 = df.grad(u1) + I
    
    f0  = patient.fiber
    s0 = None#patient.sheet
    n0 = patient.sheet_normal
    T_ref = None
    gamma = None #da.Constant(0.0)
    dev_iso_split = False
    
    active_args = (gamma, f0, s0, n0,
                   T_ref, dev_iso_split)

    for Active in [ActiveStrain, ActiveStress]:
        
        active = Active(*active_args)

        print active.type()
        
        active.Fa()
        active.Fa()

        active.Fe(F0)
        # active.Fe(F1)

        active.I1(F0)
        # active.I1(F1)

        active.I4(F0, "fiber")
        # active.I4(F1, "fiber")

        active.Wactive(F0)
        # active.Wactive(F1)

        active.get_gamma()
        active.get_activation()

        active.is_isochoric()
        
    # from IPython import embed; embed()
    exit()
