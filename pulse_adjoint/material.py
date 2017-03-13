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
from setup_optimization import RegionalParameter


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
    Base class for material
    """
    def __init__(self, T_ref = None, params = None):
        """
        Initialize base class

        :param float T_ref: Scale factor for active parameter

        """
        
        assert self._active_model in \
          ["active_stress", "active_strain", "active_strain_rossi"], \
          "The active model '{}' is not implemented.".format(self._active_model)

        if T_ref is None:
            self._T_ref = 100.0 if self._active_model == "active_stress"  else 1.0
        else:
            self._T_ref = T_ref
            
        # self._T_ref = 1.0
  

        if params:
            self.parameters = params
            for k,v in params.iteritems():
                if isinstance(v, (float, int)):
                    setattr(self, k, Constant(v))
                elif isinstance(v, RegionalParameter):
                    
                    setattr(self, k, Function(v.get_ind_space(), name = k))
                    mat = getattr(self, k)
                    mat.assign(project(v.get_function(), v.get_ind_space()))
                else:
                    setattr(self, k, v)


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

        
        # Activation
        if isinstance(self.gamma, RegionalParameter):
            # This means a regional gamma
            # Could probably make this a bit more clean
            gamma = self.gamma.get_function()
        else:
            gamma = self.gamma


        # Active stress model
        if self._active_model == 'active_stress':

            # Invariants
            I1  = self.I1(F)
            I4f = self.I4f(F)

        # Active strain model
        else:
            # Invariants
            I1  = self.I1e(F, gamma)
            I4f = self.I4fe(F, gamma)
            
        if DOLFIN_VERSION_MAJOR > 1.6:
            dim = find_geometric_dimension(F)
        else:
            dim = F.geometric_dimension()
            
        W1   = self.W_1(I1, diff = 0, dim = dim)
        W4f  = self.W_4(I4f, diff = 0)
        Wactive = self.Wactive(gamma, I4f, diff = 0)
              
        W = W1 + W4f + Wactive 
        
        return W

    def CauchyStress(self, F, p = None):
        r"""
        Chaucy Stress Tensor

        Incompressible:

        .. math::

           \sigma = \mathbf{F} \frac{\partial \Psi}{\partial \mathbf{F}} 
           - p\mathbf{I}

        Since the strain energy depends on the invariants we can write

        .. math::

           \sigma = \mathbf{F} \sum_{i = 1, i\neq3}^{N} \psi_i 
           \frac{\partial I_1}{\partial \mathbf{F}} - p\mathbf{I} 

        Compressible:

        .. math::

           \sigma = \mathbf{F} \frac{\partial \psi}{\partial \mathbf{F}}

        Since the strain energy depends on the invariants we can write

        .. math::

           \sigma = J^{-1} \mathbf{F} \sum_{i = 1}^{N} 
           \psi_i \frac{\partial I_i}{\partial \mathbf{F}}

        
        :param F: Deformation gradient
        :type F: :py:class:`dolfin.Function`
        :param p: Hydrostatic pressure
        :type p: :py:class:`dolfin.Function`

        """
        # Activation
        if isinstance(self.gamma, RegionalParameter):
            # This means a regional gamma
            # Could probably make this a bit more clean
            gamma = self.gamma.get_function()
        else:
            gamma = self.gamma

        # Left Cauchy green
        

        

        if DOLFIN_VERSION_MAJOR > 1.6:
            dim = find_geometric_dimension(F)
        else:
            dim = F.geometric_dimension()

        I = Identity(dim)
        
        # Active stress model
        if self._active_model == 'active_stress':
            B = F*F.T
            # Fibers on the current configuration
            f = F*self.f0
        
        
            # Invariants
            I1  = self.I1(F)
            I4f = self.I4f(F)

        # Active strain model
        else:
            Fa = self.Fa(gamma)
            Fe = F*inv(Fa)

            B = Fe*Fe.T

            # Fibers on the current configuration
            f = Fe*self.f0
            
            # Invariants
            I1  = self.I1e(F, gamma)
            I4f = self.I4fe(F, gamma)
            
        J = det(F)

        # The outer product of the fibers
        ff = outer(f,f)
        w1 = self.W_1(I1, diff = 1, dim = dim)
        w4f = self.W_4(I4f, diff = 1)
        wactive = self.Wactive(gamma, diff = 1)

        if p is None:
            return 2*w1*B + 2*w4f*ff  + 2*wactive*ff 
        else:
            return 2*w1*B + 2*w4f*ff  + 2*wactive*ff - p*I
        
        
    def Wactive(self, gamma, I4f = 0, diff = 0):
        """
        Acitve term in strain energy function

        :param gamma: Contraction parameter
        :type gamma: :py:class:`dolfin.Function`
        :param I4f: Quasi-invariant for fiber
        :type I4f:  :py:class:`ulf`
        :param int diff: Differentiantion number
        :returns: Value of active term
        :rtype: :py:class:`dolfin.Function` or int

        """
        
        if self._active_model == 'active_stress':

            if diff == 0:
                return self._T_ref*gamma*I4f
            elif diff == 1:
                return self._T_ref*gamma 
            
        else:
            # No active stress
            return 0

    def Fa(self, gamma):

        dim = self.f0.function_space().mesh().geometry().dim()
        if self._active_model == 'active_stress':
            return Identity(dim)
        
        else:
            f0f0 = outer(self.f0, self.f0)
        
            I = Identity(dim)
            Fa = (1-gamma)*f0f0 + pow((1-gamma), -1/float(dim-1))*(I-f0f0)
            return Fa
        
    
    def I1(self, F):
        r"""
        First Isotropic invariant

        .. math::

           I_1 = \mathrm{tr}(\mathbf{C})

        """

        C =  F.T * F
        return  tr(C)
        
        

    def I4f(self, F):
        """
        Quasi invariant in fiber direction

        .. math::

           I_{4f_0} = \mathbf{f}_0 \cdot ( \mathbf{C} \mathbf{f}_0)

        """
        if self.f0 is None:
            return Constant(0.0)

        C =  F.T * F
        return inner(C*self.f0, self.f0)

            


    def I4s(self, F):
        """
        Quasi invariant in sheet direction

        .. math::

           I_{4s_0} = \mathbf{s}_0 \cdot ( \mathbf{C} \mathbf{s}_0)

        """
        if self.s0 is None:
            return Constant(0.0)

      
        C =  F.T * F 
        return  inner(C*self.s0, self.s0)

    def I4n(self, F):
        """
        Quasi invariant in cross fiber-sheet direction

        .. math::

           I_{4n_0} = \mathbf{n}_0 \cdot ( \mathbf{C} \mathbf{n}_0)


        """
        if self.n0 is None:
            return Constant(0.0)

        C =  F.T * F
        
        return  inner(C*self.n0, self.n0)

    def I8fs(self, F):
        """
        Quasi invariant in fiber direction
        """
        if (self.f0 and self.s0) is None:
            return Constant(0.0)

        J = det(F)
        # C = pow(J, -float(2)/3) * F.T * F
        C = F.T * F
        return  inner(C*self.f0, self.s0)

    def I1e(self, F, gamma):
        r"""
        First isotropic invariant in the elastic configuration
        (active strain)

        If active stress, return the normal isotropic invariant.
        Let :math:`d` be the geometric dimension.
        If

        .. math:: 

           \mathbf{F}_a = (1 - \gamma) \mathbf{f}_0 \otimes \mathbf{f}_0  + 
           \frac{1}{\sqrt{1 - \gamma}} (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0)

        then

        .. math::

           I_1^E = I_1(1 - \gamma)^{4-d} +  
           I_{4f_0}\left(\frac{1}{(1-\gamma)^2} - (1-\gamma)^{4-d}\right) 

        If 

        .. math:: 

           \mathbf{F}_a = (1 + \gamma) \mathbf{f}_0 \otimes \mathbf{f}_0  + 
           \frac{1}{\sqrt{1 + \gamma}} (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0)

        then

        .. math::

           I_1^E = I_1(1 + \gamma)^{4-d} +  
           I_{4f_0}\left(\frac{1}{(1+\gamma)^2} - (1+\gamma)^{4-d}\right) 

        :param F: Deformation gradient
        :type F: :py:class:`dolfin.Function`
        :param gamma: Contraction parameter
        :type gamma: :py:class:`dolfin.Function`

        """

        I1  = self.I1(F)
        I4f = self.I4f(F)
        
        
        
        if DOLFIN_VERSION_MAJOR > 1.6:
            d = find_geometric_dimension(F)
        else:
            d = F.geometric_dimension()

        if self._active_model == 'active_stress':
            
            return I1
        
        # Active strain model
        else:
            
            if self._active_model == 'active_strain':
                mgamma = 1 - gamma
            
                       
            elif self._active_model == "active_strain_rossi":
                mgamma = 1+gamma
          
            return  pow(mgamma, 4-d) * I1 + (1/mgamma**2 - pow(mgamma, 4-d)) * I4f

        

    def I4fe(self, F, gamma):
        r"""
        Quasi-invariant in the elastic configuration

        If active stress, return the normal quasi-invariant.
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

        :param F: Deformation gradient
        :type F: :py:class:`dolfin.Function`
        :param gamma: Contraction parameter
        :type gamma: :py:class:`dolfin.Function`

        """

        I4f = self.I4f(F)
        

        if self._active_model == 'active_stress':
            
            return I4f
        
        # Active strain model
        else:
            if self._active_model == 'active_strain':
            
                mgamma = 1 - gamma

            
            
            elif self._active_model == "active_strain_rossi":
            
                mgamma = 1 + gamma
            
            return   1/mgamma**2 * I4f
    


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
    def __init__(self, f0 = None, gamma = None, params = None,
                 active_model = "active_strain",
                 s0 = None, n0 = None, T_ref = None):
        """
        Initialize the Holzapfel and Ogden material model

        :param f0: Fiber field
        :type f0: :py:class`dolfin.Function`
        :param gamma: Activation parameter
        :type gamma: :py:class`dolfin.Function`
        :param dict params: material parameters
        :param str active_model: The active model. Possible values are
                                 'active_stress', 'active_strain' and 
                                 'active_strain_rossi'.
        :param s0: Sheet field
        :type s0: :py:class`dolfin.Function`
        :param n0: Fiber-sheet field
        :type n0: :py:class`dolfin.Function`
        :param float T_ref: Scale factor for active parameter

        """

        # Fiber system
        self.f0 = f0
        self.s0 = s0
        self.n0 = n0
        
        self.gamma = Constant(0, name="gamma") if gamma is None else gamma

        # If no parameters are given, use the default ones
        if params is None:
            params = self.default_parameters()

        self._active_model = active_model

        
        Material.__init__(self, T_ref, params)
        


    def default_parameters(self):
        """
        Default matereial parameter for the Holzapfel Ogden model

        Taken from Table 1 row 3 of [1]
        """
        
        return {"a":2.28, "a_f":1.685, 
                "b":9.726, "b_f":15.779}


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
        
    
        


class Guccione(Material) :
    """
    Guccione material model. Copied from https://bitbucket.org/peppu/mechbench
    """
    def __init__(self, **params) :
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

        # Just some renaming in order to use existing code.
        self.gamma = self._parameters["Tactive"]
        self.f0 =  self._parameters["e1"]

    @staticmethod
    def default_parameters() :
        p = { 'C' : 2.0,
              'bf' : 8.0,
              'bt' : 2.0,
              'bfs' : 4.0,
              'e1' : None,
              'e2' : None,
              'e3' : None,
              'kappa' : None,
              'Tactive' : None }
        return p

    def is_isotropic(self) :
        """
        Return True if the material is isotropic.
        """
        p = self._parameters
        return p['bt'] == 1.0 and p['bf'] == 1.0 and p['bfs'] == 1.0



    def is_incompressible(self) :
        """
        Return True if the material is incompressible.
        """
        return self._parameters['kappa'] is None

    def strain_energy(self, F, p=None) :
        """
        UFL form of the strain energy.
        """
        params = self._parameters

        I = Identity(3)
        J = det(F)
        C = pow(J, -float(2)/3) * F.T*F
        E = 0.5*(C - I)

        CC  = Constant(params['C'], name='C')
        if self.is_isotropic() :
            # isotropic case
            Q = inner(E, E)
        else :
            # fully anisotropic
            bt  = Constant(params['bt'], name='bt')
            bf  = Constant(params['bf'], name='bf')
            bfs = Constant(params['bfs'], name='bfs')

            e1 = params['e1']
            e2 = params['e2']
            e3 = params['e3']

            E11, E12, E13 = inner(E*e1, e1), inner(E*e1, e2), inner(E*e1, e3)
            E21, E22, E23 = inner(E*e2, e1), inner(E*e2, e2), inner(E*e2, e3)
            E31, E32, E33 = inner(E*e3, e1), inner(E*e3, e2), inner(E*e3, e3)

            Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) \
              + bfs*(E12**2 + E21**2 + E13**2 + E31**2)

        # passive strain energy
        Wpassive = CC/2.0 * (exp(Q) - 1)

        # active strain energy
        if params['Tactive'] is not None :
            self.Tactive = Constant(params['Tactive'], name='Tactive')
            I4 = inner(C*e1, e1)
            # Wactive = self.Tactive/2.0 * (I4 - 1)
            Wactive = self.gamma/2.0 * (I4 - 1)
        else :
            Wactive = 0.0

        return Wpassive + Wactive 




class NeoHookean(Material):
    def __init__(self, f0 = None, gamma = None, params = None, active_model = "active_strain", s0 = None, n0 = None, T_ref = None):

        # Fiber system
        self.f0 = f0
        self.s0 = s0
        self.n0 = n0
       
        self.gamma = Constant(0, name="gamma") if gamma is None else gamma

        
        if params is None:
            params = self.default_parameters()

        for k,v in params.iteritems():
            setattr(self, k, v)

        self._active_model = active_model

        Material.__init__(self, T_ref, params)
        
    def default_parameters(self):
        return {"mu": 0.385}

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
            

   
    
