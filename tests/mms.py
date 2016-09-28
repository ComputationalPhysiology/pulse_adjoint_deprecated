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
from sympy import *
from sympy.printing import ccode
import dolfin as df
import numpy as np
import math
import pulse_adjoint.material as mat
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.compressibility import Compressibility
from pulse_adjoint.utils import QuadratureSpace
from pulse_adjoint.setup_optimization import setup_general_parameters
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({'figure.autolayout': True})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16} 

mpl.rc('font', **font)
mpl.pyplot.rc('text', usetex=True)
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True



def strain_energy_3d(F, Ta, gamma, active_model, material_model, p, a = None, b=None, a_f = None, b_f = None, mu = None):

        assert material_model in ["neo_hookean", "holzapfel_ogden"]
        
    
        C = F.transpose()*F
        I1 = C[0] + C[4] + C[8]
        I4f = C[4] # Fibers in y-direction
        I4s = C[0] # Sheets in y-direction
        I4n = C[8] # Cross Sheets in z-direction

        
        J = F[0]*(F[4]*F[8] - F[5]*F[7]) + F[1]*(F[3]*F[8] - F[5]*F[6]) + F[2]*(F[3]*F[7] - F[4]*F[6])
        
        print "I1 = ", I1
        print "I4f = ", I4f
        print "I4s = ", I4s
        

        # Find right Cauchy green
        if active_model == "active_strain":
            
            Fa_inv = Matrix([[sqrt(1-gamma), 0, 0],
                            [0, 1/(1-gamma), 0],
                            [0, 0, sqrt(1-gamma)]])
            Fe = F*Fa_inv
            Ce = Fe.transpose()*Fe
             
            I1e = Ce[0] + Ce[4] + Ce[8]
            
            
        elif active_model == "active_strain_rossi":
            
            gamma_tf = 1/sqrt(1+gamma) - 1
          
            Fa_inv = Matrix([[1- gamma_tf/(1+gamma_tf), 0, 0],
                            [0, 1-gamma/(1+gamma), 0],
                            [0, 0, 1- gamma_tf/(1+gamma_tf)]])
            
            Fe = F*Fa_inv
            Ce = Fe.transpose()*Fe

            I1e = Ce[0] + Ce[4] + Ce[8]
            I4fe = Ce[4]
            

        elif active_model == "active_stress":
            
            # C = F.transpose()*F
            I1e = I1
            I4fe = I4f
            
        else:
            raise ValueError("Unknown active model")

        # Strain energy
        if material_model == "neo_hookean":
            
            psi = 0.5*mu*(I1e - 3) - p*(J-1)
                
        else:
            
            psi = a/(2.0*b) * (exp(b*(I1e - 3)) - 1) + a/(2.0*b) * (exp(b*pow(I4f - 1, 2)) - 1) # Piecewise((0, I4f < 1), (I4f-1, True) )
            
            
        if active_model == "active_stress":
            psi += Ta*I4f 
       
        
        return psi
    


def strain_energy_2d(F, Ta, gamma, active_model, material_model, p,  a = None, b=None, a_f = None, b_f = None, mu = None):

    assert material_model in ["neo_hookean", "holzapfel_ogden"]

    # Active strain transverse fibers
    gamma_tf = 1/(1+gamma) - 1
    
    C = F.transpose()*F
    I1 = C[0] + C[3]
    I4f = C[3] # Fibers in y-direction
    I4s = C[0] # Sheets in x-direction


    # Determinant
    J = F[0]*F[3]  - F[1]*F[2]
    print "I1 = ", I1
    print "I4f = ", I4f
    print "I4s = ", I4s
        
    # Find right Cauchy green
    if active_model == "active_strain":

        Fa_inv = Matrix([[(1-gamma), 0],
                        [0, 1/(1-gamma)] ])
        Fe = F*Fa_inv
        Ce = Fe.transpose()*Fe
        I1e = Ce[0] + Ce[3]

                    
    elif active_model == "active_strain_rossi":
            
        Fa_inv = Matrix([[(1-gamma_tf/(1+gamma_tf)), 0],
                        [0, (1-gamma/(1+gamma))]])
        
        Fe = F*Fa_inv
        Ce = Fe.transpose()*Fe
        I1e = Ce[0] + Ce[3]
           

    elif active_model == "active_stress":
        I1e = I1
            
            
    else:
        raise ValueError("Unknown active model")

    
    # Strain energy
    if material_model == "neo_hookean":
            
        psi = 0.5*mu*(I1e - 2) - p*(J-1)
                
    else:
            
        psi = a/(2.0*b) * (exp(b*(I1e - 3)) - 1) + a/(2.0*b) * (exp(b*pow(I4f - 1, 2)) - 1) -p*(J-1)# Piecewise((0, I4f < 1), (I4f-1, True) )
            
            
    if active_model == "active_stress":
        psi += Ta*I4f 
        
    return psi
    

def setup_neohookean_2d(active_model, material_model):
    
    # Constant in displacement
    alpha = Symbol("alpha")  
    
    # Hydrostatic pressure
    p = Symbol("p", function = True)

    
    # Constant in strain energy
    if material_model == "neo_hookean":
    
        mu = Symbol("mu")
        matparams = {"mu":mu}

    else:
        a = Symbol("a")
        b = Symbol("b")
        a_f = Symbol("a_f")
        b_f = Symbol("b_f")
        matparams = {"a":a, "b":b, "a_f":a_f, "b_f":b_f}

    # Active strain fibers
    gamma = Symbol("gamma")

    # Active stress
    Ta = Symbol("Ta")

    # Coordinates
    X,Y = Symbol("x[0]"), Symbol("x[1]")

    # Total Deformation gradient (symbolic)
    F11, F22, F12, F21  = tuple([Symbol(s) for s in ["F11", "F22", "F12", "F21"]])
    F_sym = Matrix([[F11, F12],
                    [F21, F22]])
        
    
    U = Matrix([0.5*alpha*Y**2, 0])
    # U = Matrix([X**3*nu, 0, Z*(1.0/(3*X**2*nu + 1) - 1)])

    if active_model == "active_strain":
        p = alpha*mu*( 1 - 2*gamma + 1/((1-gamma)**2)  )*( 0.5*alpha*Y**2 + X) + mu*(1-gamma)
    else:
        p = alpha*( (mu/((1+gamma)**2)) + Ta)*( 0.5*alpha*Y**2 + X) + mu*(1+gamma)
    
    
    #Incompressible Motion
    F = Matrix([[1 + diff(U[0], X), diff(U[0], Y)],
            [diff(U[1], X), 1 + diff(U[1], Y)]])

    print "\nF = "
    print "\n\n",[ccode(e) for e in F[0:2]],
    print "\n\n",[ccode(e) for e in F[2:4]]

    # Determinant
    J = F.det()
    print "det(F) = ", J
    
    
    psi = strain_energy_2d(F, Ta, gamma, active_model, material_model, p, **matparams)
    psi_sym = strain_energy_2d(F_sym, Ta, gamma, active_model, material_model, p,**matparams)

    # First Piola - Kirchoff stress tensor
    Psym = Matrix([[diff(psi_sym, Fc).simplify() for Fc in [F11, F12]],
                [diff(psi_sym, Fc).simplify() for Fc in [F21, F22]]])

    # Chauchy stress tensor
    Tsym =  Matrix([[Fc*diff(psi_sym, Fc).simplify() for Fc in [F11, F12]],
                [Fc*diff(psi_sym, Fc).simplify() for Fc in [F21, F22]]])

   
    
    print "\nP = "
    print "\n\n",[ccode(e) for e in Psym[0:2]],
    print "\n\n",[ccode(e) for e in Psym[2:4]]

    print "F_inv_T = ",  F.inv().T

    
    convert = lambda A : A.subs(F11, F[0]).subs(F12, F[1]).subs(F21, F[2]).subs(F22, F[3])

    # f0 = Matrix([0,1])
    # fsym = F*f0
    
    # f = convert(fsym)
    # f = f / sqrt(f.dot(f))
    mesh = df.UnitSquareMesh(16,16)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    V = df.FunctionSpace(mesh, "CG", 1)
    
    outer = lambda v1, v2 : Matrix([[v1[0]*v2[0], v1[0]*v2[1]], [v1[1]*v2[0], v1[1]*v2[1]]])
    
    # Assign values to stress tensor based on analytic displacement
    P = convert(Psym)
    T = convert(Tsym)

    # Compute fiber stress
    f0 = Matrix([0,1])
    f = convert(F*f0)
    f = f / sqrt(f.dot(f))
    ff = outer(f,f)
    
    B = F * F.transpose()
    I = Matrix([[1,0],[0, 1]])

    Tsym = mu*B + 2*Ta*ff - p*I
    T = convert(Tsym)
    Tff = f.dot(T*f) 



    print "\nP = "
    print "\n\n",[ccode(e) for e in P[0:2]],
    print "\n\n",[ccode(e) for e in P[2:4]]

  
    calc_row = lambda A, n1, n2 : (diff(A[n1], X)  + diff(A[n2], Y)).simplify() 
    
    # Calculate divergence of P
    divP = Matrix([calc_row(P, 0, 1),
                   calc_row(P, 2, 3)])

    
    # Find hydrostatic pressure from equilibrium eqations
    # px = Symbol("px")
    # py = Symbol("py")
    # FinvT = F.inv().transpose()
    
    # Q = Matrix([(divP[0] - p*diff(FinvT[0], X) - px*FinvT[0] -  p*diff(FinvT[1], Y) - py*FinvT[1]).simplify(),
    #                 (divP[1] - p*diff(FinvT[2], X) - px*FinvT[2] -  p*diff(FinvT[3], Y) - py*FinvT[3]).simplify()])

    # pX = solve(Q[0], px)[0]
    # P_XY = integrate(pX, X)
    # pY = solve(Q[1].subs(px,pX), py)[0]
    # p = integrate(pY, Y)


    # Psym2 = Psym - p*J*F_sym.inv().T
    # Tff = (((alpha*Y)**2)*T[0]+ alpha*Y*T[1] + alpha*Y*T[2] + T[3])/(1+(alpha*Y)**2)
    # Tff = T[3]#/(1+(alpha*Y)**2)
    
    
    print "\nTff = ", Tff
    

    print "\ndivP = ",
    for e in divP:
        print ccode(e), ",\n"
    
    return P, divP, U, p, T, Tff


def setup_neohookean_3d(active_model, material_model):

    # Constant in displacement
    alpha = Symbol("alpha")

    # Hydrostatic pressure
    p = Symbol("p", function =True)

    # Constants in strain energy
    if material_model == "neo_hookean":
    
        mu = Symbol("mu")
        matparams = {"mu":mu}

    else:
        a = Symbol("a")
        b = Symbol("b")
        a_f = Symbol("a_f")
        b_f = Symbol("b_f")
        matparams = {"a":a, "b":b, "a_f":a_f, "b_f":b_f}

    # Active strain fibers
    gamma = Symbol("gamma")

    # Active stress
    Ta = Symbol("Ta")

    # Coordinates
    X,Y,Z = Symbol("x[0]"), Symbol("x[1]"), Symbol("x[2]")

    # Total Deformation gradient (symbolic)
    F11, F22, F12, F21, F33, F31, F13, F23, F32 = tuple([Symbol(s) for s in ["F11", "F22", "F12", "F21",
                                                                         "F33", "F31", "F13", "F23", "F32"]])
    F_sym = Matrix([[F11, F12, F13],
                [F21, F22, F23],
                [F31, F23, F33]])

    
    U = Matrix([0.5*alpha*Y**2, 0, 0])
 
    # p = 0
    # if active_model == "active_strain":
    #     p = alpha*mu*( 1 - 2*gamma + 1/((1-gamma)**2)  )*( 0.5*alpha*Y**2 + X) + mu*(1-gamma)
    # else:
    p = alpha*( (mu/((1+gamma)**2))+ Ta )*( X - 0.5*alpha*Y**2) + mu#*(1+gamma)
    # p = alpha*( (mu/((1+gamma)**2)) )*( X - 0.5*alpha*Y**2) + mu#*(1+gamma)
    p = 0
    # p = alpha*(2*Ta + mu)*( X - 0.5*alpha*Y**2) + mu
    #Incompressible Motion
    F = Matrix([[1 + diff(U[0], X), diff(U[0], Y), diff(U[0], Z)],
            [diff(U[1], X), 1 + diff(U[1], Y), diff(U[1], Z)], 
            [diff(U[2], X), diff(U[2], Y), 1 + diff(U[2], Z)]])

    def print_mat(A):
        print [ccode(e) for e in A[0:3]],
        print "\n\n",[ccode(e) for e in A[3:6]],
        print "\n\n",[ccode(e) for e in A[6:9]]
        

    print "\nF = ", print_mat(F)
    

    Ft1 = F.transpose()
    Ft = Ft1.inv()
    print "\nF.T = ", print_mat(Ft)
    
    print "\nJ = ", F.det().simplify()
    
    
    psi = strain_energy_3d(F, Ta, gamma, active_model, material_model, p, **matparams)
    psi_sym = strain_energy_3d(F_sym, Ta, gamma, active_model, material_model,p, **matparams)
   

    # First Piola - Kirchoff stress tensor
    P = Matrix([[diff(psi_sym, Fc).simplify() for Fc in [F11, F12, F13]],
                [diff(psi_sym, Fc).simplify() for Fc in [F21, F22, F23]],
                [diff(psi_sym, Fc).simplify() for Fc in [F31, F32, F33]]])

    print "\nP = ", print_mat(P)
    
    
    
    # Assign values to stress tensor based on analytic displacement
    P = P.subs(F11, F[0]).subs(F12, F[1]).subs(F13, F[2])
    P = P.subs(F21, F[3]).subs(F22, F[4]).subs(F23, F[5])
    P = P.subs(F31, F[6]).subs(F32, F[7]).subs(F33, F[8])
    # P = P.subs(Ta, 0).subs(alpha, 0).subs(gamma, 0)
    print "\nP = ", print_mat(P)
    
    calc_row = lambda A, n1, n2, n3 : (diff(A[n1], X) + diff(A[n2], Y) + diff(A[n3], Z)).simplify() 
    
    # Calculate divergence of P
    divP = Matrix([calc_row(P, 0, 1, 2),
                   calc_row(P, 3, 4, 5),
                   calc_row(P, 6, 7, 8)])

  


    print "\n\n-divP = ",
    for e in divP:
        print ccode(e), ",\n"

    
    return P, divP, U, p

    


def test_neohookean_3d():

    #Active model
    active_model = "active_stress"
    # active_model = "active_strain_rossi"
    # active_model = "active_strain"

    # Material Model
    # material_model = "holzapfel_ogden"
    material_model = "neo_hookean"

    
    # Active coefficients
    if active_model == "active_stress":
        
        gamma = df.Constant(0.0)
        Ta = df.Constant(0.9)
        
    else:
        Ta = df.Constant(0.0)
        if active_model == "active_strain_rossi":
            gamma = df.Constant(-0.3)
        else:
            gamma = df.Constant(0.2)

    # Amount of displacement
    # gamma = df.Constant(0.0)
    alpha = 0.1

    # Material coefficients
    if material_model == "neo_hookean":
        mu = 0.385
        matparams = {"mu": mu}
    else:
        a = 1.0
        b = 1.0
        a_f = 1.0
        b_f = 1.0
        matparams = {"a":a, "b":b, "a_f":a_f, "b_f":b_f}

    # Get sympy code
    P_sym, divP_sym, u_sym, p_sym = setup_neohookean_3d(active_model, material_model)


    # Convert sympy code into dolfin expressions
    P_df = df.Expression(((ccode(P_sym[0]), ccode(P_sym[1]), ccode(P_sym[2])),
                          (ccode(P_sym[3]), ccode(P_sym[4]), ccode(P_sym[5])),
                          (ccode(P_sym[6]), ccode(P_sym[7]), ccode(P_sym[8]))),
                          gamma =gamma, Ta = Ta, alpha = alpha, **matparams)

    divP_df = df.Expression((ccode(divP_sym[0]), ccode(divP_sym[1]), ccode(divP_sym[2])),
                          gamma =gamma, Ta = Ta, alpha = alpha, **matparams)

    u_df = df.Expression((ccode(u_sym[0]), ccode(u_sym[1]), ccode(u_sym[2])), alpha = alpha)
    
    p_df = df.Expression(ccode(p_sym), gamma =gamma, Ta = Ta, alpha = alpha, **matparams)


    
    DIR_BOUND = 1
    NEU_BOUND = 2


    err_u = []
    err_p = []
    err_J = []
    err_Tf = []
    ndivs = [2,4,8,12]
    # ndivs = [8]
    plot = False
    
    for ndiv in ndivs:
        mesh = df.UnitCubeMesh(ndiv, ndiv, ndiv)
        N = df.FacetNormal(mesh)

        # Mark boundaries
        ffun = df.MeshFunction("size_t", mesh, 2)
        dir_sub = df.CompiledSubDomain("(near(x[1], 0)) && on_boundary")
        ffun.set_all(NEU_BOUND)
        dir_sub.mark(ffun, DIR_BOUND)
    
    
        def make_dirichlet_bcs(W):
            bcs = [df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0, 0.0)), ffun, DIR_BOUND)]
            return bcs


        # Fibers
        V_f = V_f = QuadratureSpace(mesh, 4)
        f0 = df.interpolate(df.Expression(("0.0", "1.0", "0.0")), V_f)

        # Activation
        act = Ta if active_model == "active_stress" else gamma

   
        if material_model == "neo_hookean":
            material = mat.NeoHookean(f0, act, matparams, active_model = active_model)
        else:
            material = mat.HolzapfelOgden(f0, act, params, active_model = active_model)
   

        nsolver = "snes_solver"
        prm = {"nonlinear_solver": "snes", "snes_solver":{}}

        prm[nsolver]['absolute_tolerance'] = 1E-8
        prm[nsolver]['relative_tolerance'] = 1E-8
        prm[nsolver]['maximum_iterations'] = 15
        prm[nsolver]['linear_solver'] = 'lu'
        prm[nsolver]['error_on_nonconvergence'] = True
        prm[nsolver]['report'] = True 
    
        params= {"mesh": mesh,
                "facet_function": ffun,
                "facet_normal": N,
                "state_space": "P_2:P_1",
                "compressibility":{"type": "incompressible",
                                    "lambda":0.0},
                "material": material,
                "solve":prm, 
                "bc":{"dirichlet": make_dirichlet_bcs,
                      "body_force": -divP_df, 
                       "neumann":[[-P_df, NEU_BOUND]]}}

        df.parameters["adjoint"]["stop_annotating"] = True
        solver = LVSolver(params)
        solver.solve()

        # Get numerical solution
        uh,ph = solver.get_state().split(deepcopy=True)
        # Tff_h = df.project(solver.postprocess().fiber_stress(),df.FunctionSpace(mesh, "CG", 1))
        J = df.project(solver.postprocess().J(), df.FunctionSpace(mesh, "CG", 1))

        if plot:
            u = df.interpolate(u_df, df.VectorFunctionSpace(mesh, "CG", 2))
            p = df.interpolate(p_df, df.FunctionSpace(mesh, "CG", 1))
            # Tff = df.interpolate(Tff_df, df.FunctionSpace(mesh, "CG", 1))
            u_diff = df.project((u-uh)**2, df.FunctionSpace(mesh, "CG", 2))
        
            
            df.plot(uh, mode = "displacement", title = "u Numerical")
            df.plot(u, mode = "displacement", title = "u Exact")
            df.plot(u_diff,  title = "u diff")
            df.plot(ph, title = "p Numerical")
            df.plot(p, title = "p Exact")

            # df.plot(Tff, title = "Fiber stress exact")
            # df.plot(Tff_h, title = "Fiber stress numerical")
            # df.plot(abs(Tff-Tff_h), title = "Fiber stress diff")

            df.interactive()

        err_u.append(df.errornorm(u_df, uh, "H1", mesh = mesh))
        err_p.append(df.errornorm(p_df, ph, "L2", mesh = mesh))
        err_J.append(df.errornorm(df.Expression("1.0"), J, "L2", mesh = mesh))
        # err_Tf.append(df.errornorm(Tff_df, Tff_h, "L2", mesh = mesh))
        print "\nNdiv = ", ndiv
        print "Number of cells = ", mesh.num_cells()
        print "Error u (H1)= ", err_u[-1]
        print "Error p (L2)= ", err_p[-1]
        print "Error J (L2)= ", err_J[-1]
    
    
    fig = plt.figure()
    plt.loglog(1.0/np.array(ndivs), err_u, "b-o", label = r"$\|u - u_h\|_{H^1}$")
    plt.loglog(1.0/np.array(ndivs), err_p, "r-o", label = r"$\|p - p_h\|_{L^2}$")
    plt.loglog(1.0/np.array(ndivs), err_J, "g-o", label = r"$\|J - 1\|_{L^2}$")
    # plt.loglog(1.0/np.array(ndivs), err_Tf, "k-.o", label = r"$\|Tf - Tf_h\|_{L^2}$")
    plt.legend(loc = "best")
    # fig.savefig("test_doc/figures/mms2d_{}_qinc.pdf".format(active_model))
    plt.show()

def test_neohookean_2d():
    # Active model
    active_model = "active_stress"
    # active_model = "active_strain_rossi"
    # active_model = "active_strain"
    
    # Material Model
    # material_model = "holzapfel_ogden"
    material_model = "neo_hookean"

    
    # Active coefficients
    if active_model == "active_stress":
        
        gamma = df.Constant(0.0)
        Ta = df.Constant(0.9)
        
    else:
        Ta = df.Constant(0.0)
        if active_model == "active_strain_rossi":
            gamma = df.Constant(-0.3)
        else:
            gamma = df.Constant(0.2)

    # Amount of displacement
    alpha = 0.2
        
    # Material coefficients
    if material_model == "neo_hookean":
        mu = 0.385
        matparams = {"mu": mu}
    else:
        a = 1.0
        b = 1.0
        a_f = 1.0
        b_f = 1.0

        matparams = {"a":a, "b":b, "a_f":a_f, "b_f":b_f}

    # Get sympy code
    P_sym, divP_sym, u_sym, p_sym, T_sym, Tff_sym = setup_neohookean_2d(active_model, material_model)

    # Convert sympy code into dolfin expressions
    P_df = df.Expression(((ccode(P_sym[0]), ccode(P_sym[1]) ),
                          (ccode(P_sym[2]), ccode(P_sym[3]) )),
                         gamma =gamma, Ta = Ta, alpha = alpha, **matparams)
    T_df = df.Expression(((ccode(T_sym[0]), ccode(T_sym[1]) ),
                          (ccode(T_sym[2]), ccode(T_sym[3]) )),
                         gamma =gamma, Ta = Ta, alpha = alpha, **matparams)

    Tff_df = df.Expression(ccode(Tff_sym),
                           gamma =gamma, Ta = Ta,
                           alpha = alpha, **matparams)
        
    divP_df = df.Expression((ccode(divP_sym[0]), ccode(divP_sym[1])),
                            gamma = gamma, Ta = Ta, alpha = alpha, **matparams)
    
    u_df = df.Expression((ccode(u_sym[0]), ccode(u_sym[1])), alpha = alpha)
    p_df = df.Expression(ccode(p_sym), gamma =gamma, Ta = Ta, alpha = alpha, **matparams)
    
    
    DIR_BOUND = 1
    NEU_BOUND = 2
    
    
    err_u = []
    err_p = []
    err_J = []
    err_Tf = []
    ndivs = [2,4,8,16, 32, 64, 128]
    # ndivs = [16]
    plot = False
    
    
    for ndiv in ndivs:
        mesh = df.UnitSquareMesh(ndiv, ndiv)

        N = df.FacetNormal(mesh)

        # Mark boundaries
        ffun = df.MeshFunction("size_t", mesh, 1)
        dir_sub = df.CompiledSubDomain("near(x[1], 0)")
        ffun.set_all(NEU_BOUND)
        dir_sub.mark(ffun, DIR_BOUND)
       
    
        def make_dirichlet_bcs(W):
            bcs = [df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0)), ffun, DIR_BOUND)]
            return bcs


        # Fibers
        V_f = QuadratureSpace(mesh, 4)
        f0 = df.interpolate(df.Expression(("0.0", "1.0")), V_f)
        
        # Activation
        act = Ta if active_model == "active_stress" else gamma
    

        if material_model == "neo_hookean":
            material = mat.NeoHookean(f0, act, matparams, active_model = active_model)
        else:
            material = mat.HolzapfelOgden(f0, act, matparams, active_model = active_model)
         
        nsolver = "snes_solver"
        prm = {"nonlinear_solver": "snes", "snes_solver":{}}

        prm[nsolver]['absolute_tolerance'] = 1E-8
        prm[nsolver]['relative_tolerance'] = 1E-8
        prm[nsolver]['maximum_iterations'] = 15
        prm[nsolver]['linear_solver'] = 'lu'
        prm[nsolver]['error_on_nonconvergence'] = True
        prm[nsolver]['report'] = False


        params= {"mesh": mesh,
                     "facet_function": ffun,
                     "facet_normal": N,
                     "state_space": "P_2:P_1",
                     "compressibility":{"type": "incompressible",
                                            "lambda":0.0},
                                            "material": material,
                                            "solve":prm, 
                                            "bc":{"dirichlet": make_dirichlet_bcs,
                                                      "body_force": -divP_df,
                                                      "neumann":[[-P_df, NEU_BOUND]]}}

        df.parameters["adjoint"]["stop_annotating"] = True
        solver = LVSolver(params)
        solver.solve()

        # df.plot(df.interpolate(p_df, df.FunctionSpace(mesh, "CG", 1)), title = "Exact")
        # Get numerical solution
        uh,ph = solver.get_state().split(deepcopy=True)
        Tff_h = df.project(solver.postprocess().fiber_stress(),df.FunctionSpace(mesh, "CG", 1))
        J = df.project(solver.postprocess().J(), df.FunctionSpace(mesh, "CG", 1))
        

        if plot:
            u = df.interpolate(u_df, df.VectorFunctionSpace(mesh, "CG", 2))
            p = df.interpolate(p_df, df.FunctionSpace(mesh, "CG", 1)) 
            Tff = df.interpolate(Tff_df, df.FunctionSpace(mesh, "CG", 1))
            u_diff = df.project((u-uh)**2, df.FunctionSpace(mesh, "CG", 2))
            
        
            # df.plot(uh, mode = "displacement", title = "u Numerical")
            # df.plot(u, mode = "displacement", title = "u Exact")
            # df.plot(u_diff,  title = "u diff")
            # df.plot(ph, title = "p Numerical")
            # df.plot(p, title = "p Exact")
            # df.plot(abs(p-ph),  title = "p diff")

            df.plot(Tff, title = "Fiber stress exact")
            df.plot(Tff_h, title = "Fiber stress numerical")
            df.plot(abs(Tff-Tff_h), title = "Fiber stress diff")
            
            
            df.interactive()
            exit()

        err_u.append(df.errornorm(u_df, uh, "H1", mesh = mesh))
        err_p.append(df.errornorm(p_df, ph, "L2", mesh = mesh))
        err_J.append(df.errornorm(df.Expression("1.0"), J, "L2", mesh = mesh))
        err_Tf.append(df.errornorm(Tff_df, Tff_h, "L2", mesh = mesh))
        print "\nNdiv = ", ndiv
        print "Number of cells = ", mesh.num_cells()
        print "Error u (H1)= ", err_u[-1]
        print "Error p (L2)= ", err_p[-1]
        print "Error J (L2)= ", err_J[-1]
    
    
    fig = plt.figure()
    h  = 1.0/np.array(ndivs)
    plt.loglog(h, err_u, "b-o", label = r"$\|u - u_h\|_{H^1}$")
    plt.loglog(h, err_p, "r-o", label = r"$\|p - p_h\|_{L^2}$")
    plt.loglog(h, err_J, "g-o", label = r"$\|J - 1\|_{L^2}$")
    plt.loglog(h, err_Tf, "c-o", label = r"$\|Tf - Tf_h\|_{L^2}$")
    plt.loglog(h, 3e-3*h**2, "k-.o", label = r"$Ch^{2}$")
    plt.loglog(h, 0.5e-3*h**(2.6), "k-.x", label = r"$Ch^{2.6}$")
    # plt.loglog(h, h**(2.5), "k-.o", label = r"$Ch^{2.5}$")
    # plt.loglog(h, h**(3.5), "k-.o", label = r"$Ch^{3.5}$")
    plt.legend(loc = "best")
    fig.savefig("test_doc/figures/mms2d_{}.pdf".format(active_model))
    plt.show()

if __name__ == "__main__":
    setup_general_parameters()
    # setup_neohookean_2d()
    # setup_neohookean_3d()
    # test_neohookean_3d()
    test_neohookean_2d()
