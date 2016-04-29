from dolfin import *
from dolfin_adjoint import *

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
        

class Compressibility(object):

    class Incompressible(object):
        def __init__(self, parameters):
            mesh = parameters["mesh"]
            
            V_str, Q_str = ("P_2", "P_1") if not parameters.has_key("state_space") \
              else parameters["state_space"].split(":")

            # Displacemet Space
            V = VectorFunctionSpace(mesh, V_str.split("_")[0], 
                                    int(V_str.split("_")[1]))

            # Lagrange Multiplier
            Q = FunctionSpace(mesh, Q_str.split("_")[0], 
                              int(Q_str.split("_")[1]))
            self.W = V*Q
            self.w = Function(self.W, name = "displacement-pressure")
            self.w_test = TestFunction(self.W)
            self.u_test, self.p_test = split(self.w_test)
            self.u, self.p = split(self.w)
    
        def __call__(self, J):
            return (J - 1)*self.p
        
        def get_displacement_space(self):
            return self.W.sub(0)
        
        def get_displacement_variable(self):
            return self.u
        
        def get_displacement(self, name, annotate = True):
            D = self.get_displacement_space()
            V = D.collapse()
        
            fa = FunctionAssigner(V, D)
            u = Function(V, name = name)
            fa.assign(u, self.w.split()[0], 
                      annotate = annotate)
            return u
        
        def calculate_average_volume_ratio(self, w, mesh):
            return assemble(det(grad(split(w)[0]) \
                                + Identity(3))*dx) \
                                /assemble(1.0*Measure("dx", domain = mesh))
        
        def calculate_max_volume_ratio(self, w, mesh):
            J = project(det(grad(split(w)[0]) + Identity(3)), 
                        self.W.sub(1).collapse(), name = "J max")

            return max(abs(J.vector().array() -1))
    
    class StabalizedIncompressible(Incompressible):
        """
        Formulation from Sander Land 2015 "Improving the Stability of Cardiac
        Mechanical Simulations"
        """
        def __init__(self, parameters):
            super(type(self), self).__init__(parameters)

            if parameters["compressibility"].has_key("lambda"):
                self.lamda = Constant(parameters["compressibility"]["lambda"], name = "incomp_penalty")
            else:
                print "Warning: Lambda is not provided. Use Incompressible model"
                self.lamda = Constant(0.0,name = "incomp_penalty")
              
                                  

        
        def __call__(self, J):
            return (J - 1)*self.p + 0.5*self.lamda*(J - 1)**2
       
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
        
        def get_displacement(self, name, annotate = True):
            return Function(self.w, name = name)
        
        def __call__(self, J):
            return self.lamda*(J - 1)**2
    
        def calculate_average_volume_ratio(self, w, mesh):
            return assemble(det(grad(w) + Identity(3))*dx)/assemble(1.0*Measure("dx", domain = mesh))
    
        def calculate_max_volume_ratio(self, w, mesh):
            J = project(det(grad(w) + Identity(3)), FunctionSpace(mesh, "DG", 0))
            return max(abs(J.vector().array() - 1))
    
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
        
        def calculate_average_volume_ratio(self, w, mesh):
            return assemble(det(grad(split(w)[0]) + Identity(3))*dx)/assemble(1.0*Measure("dx", domain = mesh))
        
        def calculate_max_volume_ratio(self, w, mesh):
            J = project(det(grad(split(w)[0]) + Identity(3)), self.W.sub(1).collapse(), name = "max_volume_ration")
            return max(abs(J.vector().array() - 1))
        
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
