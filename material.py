#!/usr/bin/env python
# Copyright (C) 2016 Henrik Finsberg
#
# This file is part of CAMPASS, but based on PULSE.
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


    def strain_energy(self, F):
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




class Guccione(object):
    """FIXME:"""
    def __init__(self, **params):
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

    @staticmethod
    def default_parameters():
        p = { 'C': 2.0,
              'bf': 8.0,
              'bt': 2.0,
              'bfs': 4.0,
              'e1': None,
              'e2': None,
              'e3': None,
              'Tactive': None }
        return p

    def is_isotropic(self):
        """
        Return True if the material is isotropic.
        """
        p = self._parameters
        return p['bt'] == 1.0 and p['bf'] == 1.0 and p['bfs'] == 1.0

    def is_incompressible(self):
        """
        Return True if the material is incompressible.
        """
        return self._parameters['kappa'] is None

    def strain_energy(self, F):
        """
        UFL form of the strain energy.
        """
        params = self._parameters

        I = Identity(3)
        J = det(F)
        C = pow(J, -float(2)/3) * F.T*F
        E = 0.5*(C - I)

        CC  = Constant(params['C'], name='C')
        if self.is_isotropic():
            # isotropic case
            Q = inner(E, E)
        else:
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
        if params['Tactive'] is not None:
            self.Tactive = Constant(params['Tactive'], name='Tactive')
            I4 = inner(C*e1, e1)
            Wactive = self.Tactive/2.0 * (I4 - 1)
        else:
            Wactive = 0.0
        
        return Wpassive + Wactive

    def set_active_stress(self, value):
        self.Tactive.assign(value)

    def get_active_stress(self):
        return float(self.Tactive)

