#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
from .active import *
from .compressibility import compressibility

from ..dolfinimport import *
from ..setup_optimization import RegionalParameter
from ..utils import get_dimesion
from ..adjoint_contraction_args import logger



def get_dimesion(F):
    
    if DOLFIN_VERSION_MAJOR > 1.6:
        dim = find_geometric_dimension(F)
    else:
        dim = F.geometric_dimension()

    return dim

def subplus(x):
    r"""
    Ramp function

    .. math::

       \max\{x,0\}

    """
    
    return conditional(ge(x, 0.0), x, 0.0)

def heaviside(x):
    r"""
    Heaviside function

    .. math::

       \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

    """
    
    return conditional(ge(x, 0.0), 1.0, 0.0)

class Material(object):
    """
    Initialize material model
    
    Parameters
    ----------

    f0 : :py:class`dolfin.Function`
        Fiber field
    gamma : :py:class`dolfin.Function`
        Activation parameter
    params : dict
        Material parameters
    active_model : str
        Active model - active strain or active stress
    s0 : :py:class`dolfin.Function`
        Sheets
    n0 : :py:class`dolfin.Function`
        Sheet - normal
    T_ref : float
        Scale factor for activation parameter (default = 1.0)
    dev_iso_split : bool
        Decouple deformation into deviatoric and isochoric deformations
    eta : float
        Fraction of transverse active tesion for active stress formulation.
        0 = active only along fiber, 1 = equal forces in all directions (default=0.0).
    """
    def __init__(self, f0 = None, gamma = None, params = None,
                 active_model = "active_strain", s0 = None,
                 n0 = None, T_ref = None, dev_iso_split = True,
                 compressible_model = "incompressible",
                 eta = 0.0, *args, **kwargs):


        # Parameters
        if params is None:
            params = self.default_parameters()

        for k,v in params.iteritems():
            setattr(self, k, v)

                
        self.parameters = params
        for k,v in params.iteritems():
                
            if isinstance(v, (float, int)):
                setattr(self, k, Constant(v))
                
            elif isinstance(v, RegionalParameter):
                   
                setattr(self, k, Function(v.get_ind_space(), name = k))
                mat = getattr(self, k)
                matfun = v.get_function()
                ind_space = v.get_ind_space()
                mat.assign(project(matfun, ind_space))
                
            else:
                setattr(self, k, v)
                    

        # Active model
        assert active_model in \
            ["active_stress", "active_strain", "active_strain_rossi"], \
            "The active model '{}' is not implemented.".format(active_model)
        
        active_args = (gamma, f0, s0, n0,
                       T_ref, dev_iso_split)
        # Activation
        if active_model == "active_stress":
            self.active = ActiveStress(*active_args, eta = eta)
        else:
            self.active = ActiveStrain(*active_args)

        self._compressible_model = compressible_model

    def compressible_model(self):
        return self._compressible_model

    def compressibility(self, p, J):

        model = self.compressible_model()
        return compressibility(model, p, J)

    def get_active_model(self):
        return self.active.get_model_type()

    def get_material_model(self):
        return self._model

    def is_isochoric(self):
        return self.active.is_isochoric()

    def get_gamma(self):
        """
        Return the contraciton paramter.
        If regional, this will return one parameter
        for each segment.
        """
        return self.active.get_gamma()
    def get_activation(self):
        """
        Return the contraciton paramter.
        If regional, this will return a piecewise
        constant function (DG_0)
        """
        return self.active.get_activation()
    def get_component(self, component="fiber"):
        return self.active.get_component(component)

    def strain_energy(self, F):
        r"""
        Strain-energy density function.

        .. math::
        
           \mathcal{W} = \mathcal{W}_1 + \mathcal{W}_{4f}
           + \mathcal{W}_{\mathrm{active}}

        where 

        .. math::

           \mathcal{W}_{\mathrm{active}} = 
           \begin{cases} 
             0 & \text{if acitve strain} \\
             \gamma I_{4f} & \text{if active stress}
           \end{cases}


        :param F: Deformation gradient
        :type F: :py:class:`dolfin.Function`

        """
              

        # Invariants
        I1  = self.active.I1(F)
        I4f = self.active.I4(F)

        # Active stress
        Wactive = self.active.Wactive(F, diff = 0)
        
        dim = get_dimesion(F)
        W1   = self.W_1(I1, diff = 0, dim = dim)
        W4f  = self.W_4(I4f, diff = 0)
        
              
        W = W1 + W4f + Wactive 
        
        return W

    



    def CauchyStress(self, F, p=None, deviatoric = False):

        I = Identity(3)


        F = variable(F)

        if p is None:
            psi = self.strain_energy(F)
        else:
            J = det(F)
            psi = self.strain_energy(F) - p*(J-1)


       

        fiber = self.active.get_component("fiber")
        sheet = self.active.get_component("sheet")
        cross_sheet = self.active.get_component("sheet_normal")


        # First Piola Kirchoff
        S = self.SecondPiolaStress(F,p)
        P = F*S
                
        # Cauchy stress
        T = InversePiolaTransform(P, F)
        

        if deviatoric:
            logger.debug("Return deviatoric Cauchy stress")
            return (T - (1.0/3.0) * tr(T)*I )

        else:
            logger.debug("Return total Cauchy stress")
            return T

    
    def SecondPiolaStress(self, F, p, *args, **kwargs):

        dim = get_dimesion(F)
        I = Identity(dim)

        f0 = self.active.get_component("fiber")
        f0f0 = outer(f0,f0)
        
        I1  = variable(self.active.I1(F))
        I4f = variable(self.active.I4(F))

        Fe = self.active.Fe(F)
        Ce = Fe.T*Fe

        # Elastic volume ratio
        J = variable(det(Fe))
        

        w1   = self.W_1(I1, diff = 1, dim = dim)
        w4f  = self.W_4(I4f, diff = 1)

        
        
        # Total Stress
        S_bar = 2 * w1*I + 2 * w4f * f0f0 

        if self.is_isochoric():

            # Deviatoric
            Dev_S_bar = S_bar - (1.0/3.0)*inner(S_bar, Ce)*inv(Ce)

            S_mat = J**(-2.0/3.0)*Dev_S_bar
        else:
            S_mat = S_bar
            

        # Volumetric
        psi_vol = self.compressibility(p,J)
        S_vol = J*diff(psi_vol, J)*inv(Ce)

        

        # Active stress
        wactive = self.active.Wactive(F, diff = 1)
        eta = self.active.eta()
        
        S_active = wactive * ( f0f0 + eta * (I - f0f0))
        

        S = S_mat + S_vol + S_active
        
      
        return S
        
        

    def FirstPiolaStress(self, F, p=None, *args, **kwargs):

        I = Identity(3)
        
        F = variable(F)        
        # First Piola Kirchoff
        psi_iso = self.strain_energy(F)
        P_iso = diff(psi_iso, F)
        
        J = variable(det(F))
        psi_vol = self.compressibility(p,J)
        P_vol = J*diff(psi_vol, J)*inv(F).T

        P = P_iso + P_vol
        
        return P
        
class HolzapfelOgden(Material):
    r"""
    Transversally isotropic version of the
    Holzapfel and Ogden material model

    .. math::

       \mathcal{W}(I_1, I_{4f_0})  
       = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
       + \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)

    where 

    .. math::

       (\cdot)_+ = \max\{x,0\}


    .. rubric:: Reference

    [1] Holzapfel, Gerhard A., and Ray W. Ogden.  "Constitutive modelling of 
    passive myocardium: a structurally based framework for material characterization.
    "Philosophical Transactions of the Royal Society of London A: 
    Mathematical, Physical and Engineering Sciences 367.1902 (2009): 3445-3475.
    

    """
    _model = "holzapfel_ogden"
    @staticmethod
    def default_parameters():
        """
        Default matereial parameter for the Holzapfel Ogden model

        Taken from Table 1 row 3 of [1]
        """
        
        return {"a":2.28, "a_f":1.685, 
                "b":9.726, "b_f":15.779}



            

        
        

    # def CauchyStress(self, F, p=None, deviatoric=False):

    #     I1  = self.active.I1(F)
    #     I4f = self.active.I4(F)

    #     # Active stress
    #     wactive = self.active.Wactive(F, diff = 1)
        
    #     dim = get_dimesion(F)
    #     I = Identity(dim)
    #     w1   = self.W_1(I1, diff = 1, dim = dim)
    #     w4f  = self.W_4(I4f, diff = 1)

        
    #     Fe = self.active.Fe(F)
    #     Be = Fe*Fe.T
    #     B = F*F.T

    #     fe = Fe*self.active.get_component("fiber")
    #     f = F*self.active.get_component("fiber")

    #     fefe = outer(fe, fe)
    #     ff = outer(f,f)
        
        
              
    #     # T = 2*w1*Be + 2*w4f*fefe + wactive*ff
    #     # T = 2*w4f*ff + wactive*ff  #2*w1*B
    #     T = wactive*ff 
    #     if deviatoric:
    #         return T - tr(T)*I
        
        
    #     if p is None:
    #         return T

    #     else:
    #         return T - p*I
            
        
        
    def W_1(self, I_1, diff=0, *args, **kwargs):
        r"""
        Isotropic contribution.

        If `diff = 0`, return

        .. math::

           \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)

        If `diff = 1`, return

        .. math::

           \frac{a}{b} e^{ b (I_1 - 3)} 

        If `diff = 2`, return

        .. math::

           \frac{a b}{2}  e^{ b (I_1 - 3)}     
        
        """

       
        a = self.a
        b = self.b

        if diff == 0:
            return a/(2.0*b) * (exp(b*(I_1 - 3)) - 1)
        elif diff == 1:
            return a/2.0 * exp(b*(I_1 - 3))
        elif diff == 2:
            return a*b/2.0 * exp(b * (I_1 - 3))

    def W_4(self, I_4, diff=0, *args, **kwargs):
        r"""
        Anisotropic contribution.

        If `diff = 0`, return

        .. math::

           \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)

        If `diff = 1`, return

        .. math::

           a_f (I_{4f_0} - 1)_+ e^{ b_f (I_{4f_0} - 1)^2} 

        If `diff = 2`, return

        .. math::

           a_f h(I_{4f_0} - 1) (1 + 2b(I_{4f_0} - 1)) e^{ b_f (I_{4f_0} - 1)_+^2} 

        where
        
        .. math::

           h(x) = \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

        is the Heaviside function.
        
        """
        a = self.a_f
        b = self.b_f

        if I_4 == 0:
            return 0

        if diff == 0:
            return a/(2.0*b) * heaviside(I_4 - 1) * (exp(b*pow(I_4 - 1, 2)) - 1)
        
        elif diff == 1:
            return a * subplus(I_4 - 1) \
                     * exp(b * pow(I_4 - 1, 2))
        elif diff == 2:
            return a * heaviside(I_4 - 1) \
                     * (1 + 2.0 * b * pow(I_4 - 1, 2)) \
                     * exp(b * pow(I_4 - 1, 2))
        
    
        


class Guccione(Material) :
    """
    Guccione material model. 

    .. note: 
       
        Only implemented for active stress model


    """
    _model = "guccione"
    @staticmethod
    def default_parameters() :
        p = { 'C' : 2.0,
              'bf' : 8.0,
              'bt' : 2.0,
              'bfs' : 4.0 }
        return p

    def SecondPiolaStress(self, F, p, *args, **kwargs):

        P = self.FirstPiolaStress(F, p)
        S = inv(F)*P
        return S
        

        
    def is_isotropic(self) :
        """
        Return True if the material is isotropic.
        """
        
        p = self.parameters
        return p['bt'] == 1.0 and p['bf'] == 1.0 and p['bfs'] == 1.0


    def strain_energy(self, F_) :
        """
        UFL form of the strain energy.
        """        
        params = self.parameters

        # Elastic part of deformation gradient
        F = self.active.Fe(F_)
        C = F.T*F

        I = Identity(3)
        J = det(F)
        dim = get_dimesion(F)
        
        if self.is_isochoric():
            C_bar = pow(J, -float(2)/dim)*C
            E = 0.5*(C_bar - I)
            
        else:
            
            E = 0.5*(C - I)

        
        CC  = Constant(params['C'], name='C')
        
        e1 = self.active.get_component("fiber")
        e2 = self.active.get_component("sheet")
        e3 = self.active.get_component("sheet_normal")
        
        if self.is_isotropic() :
            # isotropic case
            Q = inner(E, E)
        else :
            # fully anisotropic
            bt  = Constant(params['bt'], name='bt')
            bf  = Constant(params['bf'], name='bf')
            bfs = Constant(params['bfs'], name='bfs')

            E11, E12, E13 = inner(E*e1, e1), inner(E*e1, e2), inner(E*e1, e3)
            E21, E22, E23 = inner(E*e2, e1), inner(E*e2, e2), inner(E*e2, e3)
            E31, E32, E33 = inner(E*e3, e1), inner(E*e3, e2), inner(E*e3, e3)

            Q = bf*E11**2 + bt*(E22**2 + E33**2 + 2*E23**2 ) \
                + bfs*(2*E12**2 + 2*E13**2 )
            
        # passive strain energy
        Wpassive = CC/2.0 * (exp(Q) - 1)
        Wactive = self.active.Wactive(F, diff = 0)        
        
        return Wpassive + Wactive 

    
class NeoHookean(Material):
    """
    Class for Neo Hookean material
    """
    _model = "neo_hookean"

    @staticmethod
    def default_parameters():
        return {"mu": 15.0}

    def W_1(self, I_1, diff = 0, dim = 3, *args, **kwargs):
        
        mu = self.mu

        if diff == 0:
            return  0.5*mu*(I_1-dim)
        elif diff == 1:
            return 0.5*mu
        elif diff == 2:
            return 0
        
    def W_4(self, *args, **kwargs):
        return 0


class LinearElastic(Material):
    """
    Class for linear elastic material
    """
    _model = "linear_elastic"

    @staticmethod
    def default_parameters():
        return {"mu": 100.0,
                "lmbda": 1.0}

    def strain_energy(self, F_):


        F = self.active.Fe(F_)

        dim = get_dimesion(F)
        gradu = F - Identity(dim)
        epsilon = Constant(0.5) * (gradu + gradu.T)
        W = self.lmbda/2*(tr(epsilon)**2) + self.mu*tr(epsilon*epsilon)

        # Active stress
        Wactive = self.active.Wactive(F, diff = 0)
        
        return W + Wactive

class StVenantKirchhoff(Material):
    """
    Class for linear elastic material
    """
    _model = "saint_venant_kirchhoff"

    @staticmethod
    def default_parameters():
        return {"mu": 300.0,
                "lmbda": 1.0}

    def strain_energy(self, F_):

        F = self.active.Fe(F_)
        
        dim = get_dimesion(F)

        I = Identity(3)
        J = det(F)
        dim = get_dimesion(F)
        if self.active.is_isochoric():
            F_bar = pow(J, -float(1)/dim)*F
        else:
            F_bar = F

        
        C_bar = F_bar.T*F_bar
        E = 0.5*(C_bar - I)
        
        W = self.lmbda/2*(tr(E)**2) + self.mu*tr(E*E)

        # Active stress
        Wactive = self.active.Wactive(F, diff = 0)
        
        return W + Wactive


if __name__ == "__main__":

    from patient_data import LVTestPatient
    patient = LVTestPatient()

    from setup_parameters import (setup_adjoint_contraction_parameters,
                                  setup_material_parameters, setup_general_parameters)
    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    params["phase"] == "all"
    active_model = "active_stress"
    params["active_model"] = active_model
    params["T_ref"] = 1.0

    material_model = "holzapfel_ogden"
    # material_model = "guccione"
    # material_model = "neo_hookean"

    from setup_optimization import make_solver_params
    solver_parameters, pressure, paramvec= make_solver_params(params, patient)
    
    gamma = Constant(100.0)

    matparams = setup_material_parameters(material_model)
    matparams["a_f"] = 0.0
    args = (patient.fiber,
            gamma,
            matparams,
            active_model,
            patient.sheet,
            patient.sheet_normal,
            params["T_ref"])

    if material_model == "holzapfel_ogden":
        material = HolzapfelOgden(*args)

    elif material_model == "guccione":
        material = Guccione(*args)
        
    elif material_model == "neo_hookean":
        material = NeoHookean(*args)


    from IPython import embed; embed()
    exit()
    # print "Is isochoric: ", material.is_isochoric()
    # print assemble( material.strain_energy(u0) * dx)
    # assert assemble( material.strain_energy(u0) * dx) < 1e-10
    # print assemble( material.strain_energy(u1) * dx)

    V = VectorFunctionSpace(patient.mesh, "CG", 2)
    u0 = Function(V)
    # u1 = Function(V, "../tests/data/inflate_mesh_simple_1.xml")
    # F = grad(u1) + Identity(3)
    F = grad(u0) + Identity(3)
    T = material.CauchyStress(F)
    f0 = patient.fiber
    f = F*f0
    

    # f0 = patient.fiber
    # F = DeformationGradient(u1)
    # f = F*f0
    
    # T = material.CauchyStress(u1)
    # W = FunctionSpace(patient.mesh, "CG", 1)
    meshvol = assemble(Constant(1.0)*dx(domain=patient.mesh))
    Tf = assemble( inner(T*f/f**2, f) * dx) / meshvol
    print Tf
    # from IPython import embed; embed()
    exit()

    
    # print assemble( inner(T, e) * dx)
    
    # solver_parameters["material"] = material


    # from lvsolver import LVSolver
    # solver = LVSolver(solver_parameters)
    # solver.parameters["solve"]["snes_solver"]["report"] = True

    
    # solver.solve()
    

    # pressure["p_lv"].t = 0.1

    # solver.solve()


   
    
