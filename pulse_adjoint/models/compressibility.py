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
from ..dolfinimport import *
from ..adjoint_contraction_args import logger

def get_compressibility(parameters):

    if not parameters.has_key("compressibility"):
        return Compressibility.Incompressible(parameters)

    assert parameters["compressibility"].has_key("type")

    assert parameters["compressibility"]["type"] in \
      ["incompressible", "stabalized_incompressible", "penalty", "hu_washizu"]

    if parameters["compressibility"]["type"] == "incompressible":
        return Compressibility.Incompressible(parameters)

    elif parameters["compressibility"]["type"] == "stabalized_incompressible":
        return Compressibility.StabalizedIncompressible(parameters)

    elif parameters["compressibility"]["type"] == "penalty":
        return Compressibility.Penalty(parameters)
    
    elif parameters["compressibility"]["type"] == "hu_washizu":
        return Compressibility.HuWashizu(parameters)


# class CardiacMechanicsProblem(object):


def compressibility(model, *args, **kwargs):

    if model == "incompressible":
        return incompressible(*args, **kwargs)


def incompressible(p,J):
    return -p*(J-1.0)


        
    
        

class Compressibility(object):

    class Incompressible(object):

        
        def __init__(self, parameters):
            
            mesh = parameters["mesh"]
            
            element = "taylor_hood" if not parameters.has_key("element_type") \
                       else parameters["element_type"]

            msg  = "Supported elements are 'taylor_hood' and 'mini'"
            assert element in ["taylor_hood", "mini"], msg

            if DOLFIN_VERSION_MAJOR > 1.6:

                if element == "taylor_hood":
                    
                    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
                    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

                    self._u_space = FunctionSpace(mesh, P2)
                    self._p_space = FunctionSpace(mesh, P1)
                    self.W = FunctionSpace(mesh, P2*P1)
                else:

                    logger.warning("MINI elements are experimental. "
                                   "Things might fail.")
                    bdim = 3 if mesh.ufl_domain().topological_dimension() == 2 else 4
                    
                    P1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
                    B = VectorElement("Bubble",   mesh.ufl_cell(), bdim)
                    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
               
                    self.W = FunctionSpace(mesh, (P1 + B) * Q)
                    
     
                
                
            else:

                if element == "mini":
                    logger.warning("mini elements are not supported "
                                   "for this version of fenics. "
                                   "Consider upgrading to version 2016.1.0")
                P2 = VectorFunctionSpace(mesh, "Lagrange", 2)
                P1 = FunctionSpace(mesh, "Lagrange", 1)
                self._u_space = P2
                self._p_space = P1
                self.W = P2*P1

            
            self.w = Function(self.W, name = "displacement-pressure")
            self.w_test = TestFunction(self.W)
            self.u_test, self.p_test = split(self.w_test)
            self.u, self.p = split(self.w)
    

        def is_incompressible(self):
            return True

        def get_state_space(self):
            return self.W

        def get_state(self):
            return self.w

        def get_state_test(self):
            return self.w_test
        
        def get_displacement_space(self):
            return self.W.sub(0)
        
        def get_u_space(self):
            return self._u_space
        def get_p_space(self):
            return self._p_space

        def get_pressure_space(self):
            return self.W.sub(1)
        
        def get_displacement_variable(self):
            return self.u
        def get_lp_variable(self):
            return self.p
        
        def get_displacement(self, name, annotate = True):
            D = self.get_displacement_space()
            V = D.collapse()
        
            fa = FunctionAssigner(V, D)
            u = Function(V, name = name)
            fa.assign(u, self.w.split()[0], 
                      annotate = annotate)
            return u

        def get_hydrostatic_pressue(self, name = "p", annotate = True):
            D = self.get_pressure_space()
            V = D.collapse()
        
            fa = FunctionAssigner(V, D)
            p = Function(V, name = name)
            fa.assign(p, self.w.split()[1], 
                      annotate = annotate)
            return p
        

    
    class StabalizedIncompressible(Incompressible):
        """
        Formulation from Sander Land 2015 "Improving the Stability of Cardiac
        Mechanical Simulations"
        """
        def __init__(self, parameters):
            super(type(self), self).init_spaces(parameters)

            if parameters["compressibility"].has_key("lambda"):
                self.lamda = Constant(parameters["compressibility"]["lambda"], name = "incomp_penalty")
            else:
                print "Warning: Lambda is not provided. Use Incompressible model"
                self.lamda = Constant(0.0,name = "incomp_penalty")
              
                                  

        
        def __call__(self, J):
            return (J - 1)*self.p + 0.5*self.lamda*(J - 1)**2

        def is_incompressible(self):
            return True
       
    class Penalty(object):
        def __init__(self, parameters):
            mesh = parameters["mesh"]
            self.W = VectorFunctionSpace(mesh, "CG", 1)
            self.w = Function(self.W, name = "displacement")    
            self.lamda = Constant(parameters["compressibility"]["lambda"],
                                  name = "incomp_penalty")
            self.w_test = TestFunction(self.W)
            self.u_test = self.w_test
            
        def get_displacement_space(self):
            return self.W
        
        def get_displacement_variable(self):
            return self.w

        def get_hydrostatic_pressue(self, *args):
            return None
        
        def get_displacement(self, name, annotate = True):
            V = self.get_displacement_space()
            u = Function(V, name = name)
            u.assign( self.w,annotate = annotate)
            return u
        
        def __call__(self, J):
            return self.lamda * (J**2 - 1 - 2*ln(J))
            # return self.lamda*( 0.5*(J - 1)**2 - ln(J) ) 
            # return self.lamda*(J - 1)**2

        def is_incompressible(self):
            return False

      
    class HuWashizu(object):
        """
        This gives the formulation used in Goektepe et al 2011
        'Computational modeling of passive myocardium'.
        """
        
        def __init__(self, parameters):
            mesh = parameters["mesh"]
            V = VectorFunctionSpace(mesh, "CG", 1)
            self.Q = FunctionSpace(mesh, "DG", 0)
            self.W = MixedFunctionSpace([V, self.Q, self.Q])
            self.lamda = parameters["compressibility"]["lambda"]
            self.w_test = TestFunction(self.W)
            self.u_test, self.p_test, self.d_test = split(self.w_test)
            
            self.w = Function(self.W, name = "displacement-dilatation-pressure")
            
            u,p,d = self.w.split()

            #Set dilatation field to 1 to avoid blow up.
            fa = FunctionAssigner(self.W.sub(2), self.Q)
            fa.assign(d, interpolate(Constant(1.0), self.Q))
            
            self.u, self.p, self.d = split(self.w)
        
        def __call__(self, J):
            return (J - self.d)*self.p + self.lamda*ln(self.d)**2
        
        def is_incompressible(self):
            raise NotImplementedError

        def get_displacement_space(self):
            return self.W.sub(0)
        
        def get_displacement_variable(self):
            return self.u
        
        def get_displacement(self, name, annotate = True):
            D = self.get_displacement_space()
            V = D.collapse()
        
            fa = FunctionAssigner(V, D)
            u = Function(V, name = name)
            fa.assign(u, self.w.split()[0], annotate)
            return u

    #class AugmentedHuWashizu(HuWashizu):
    #    """
    #    Hu Washizu with an augmented Lagrange multiplier.
    #    """
    #    
    #    def __init__(self, parameters):
    #        super(self.__class__, self).__init__(parameters)
    #        self.K = Function(self.Q, name = "Lagrange Multiplier")
    #    
    #    def __call__(self, J):
    #        return super(self.__class__, self).__call__(J) + self.K*(self.d - 1)
    #    
    #    def updated_multiplier(self):
    #        self.K.vector()[:] += project(self.lamda*(self.d - 1), self.Q).vector()[:]
    #
    #    def calculate_average_multiplier(self, w, mesh):
    #        return assemble(self.K*dx)/assemble(1.0*Measure("dx", domain = mesh))
    #    
    #    def calculate_max_multiplier(self, w, mesh):
    #        return max(self.K.vector().array())
