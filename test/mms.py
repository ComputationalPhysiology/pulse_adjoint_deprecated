import dolfin as df
from pulse_adjoint.models import material as mat
from pulse_adjoint.lvsolver import LVSolver

df.parameters["form_compiler"]["quadrature_degree"] = 4


# active_model = "active_stress"
active_model = "active_strain"


alpha = df.Constant(0.3)
mu = df.Constant(0.385)

if active_model == "active_stress":
    gamma = df.Constant(0.0)
    Ta = df.Constant(0.9)
    act = Ta
else:
    gamma = df.Constant(0.3)
    Ta = df.Constant(0.0)
    act = gamma


# Fiber
f0_expr = df.Expression(("0.0", "1.0"), degree=1)

# Exact solution
u_exact = df.Expression(("0.5*alpha*x[1]*x[1]", "0.0"), alpha = alpha, degree=2)

p_exact = df.Expression("alpha*( 0.5*alpha*x[1]*x[1] + x[0])*(mu / ((1-gamma)*(1-gamma)) + Ta)",
                        alpha = alpha, mu = mu, gamma = gamma, Ta = Ta, degree=2)

F_exact = df.Expression((("1.0","alpha*x[1]"),
                         ("0.0", "1.0")), degree = 1, alpha = alpha)
Ff0f0 = df.Expression((("0.0","alpha*x[1]"),
                       ("0.0", "1.0")), degree = 1, alpha = alpha)
FinvT = df.Expression((("1.0","0.0"),
                       ("-alpha*x[1]", "1.0")), degree = 1, alpha = alpha)

mgamma = (1-gamma)
d = 2

P = mu* pow(mgamma, 4-d) * F_exact  \
    + (Ta + mu*( 1.0/mgamma**2 - pow(mgamma, 4-d))) * Ff0f0  \
    - p_exact*FinvT

T = -P*F_exact.T

matparams = {"mu":mu}




err_u = []
err_p = []
err_J = []

ndivs = [2,4,8,16, 32, 64, 128][:-2]
# ndivs = [32]
for N in ndivs:


    mesh = df.UnitSquareMesh(N,N)

    
    dg1_fel = df.FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 1)
    DG1 = df.FunctionSpace(mesh, dg1_fel)
    P1_fel = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P1 = df.FunctionSpace(mesh, P1_fel)
    
    P2_vel = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P2 = df.FunctionSpace(mesh, P2_vel)
    
    quad_vel = df.VectorElement("Quadrature", mesh.ufl_cell(),
                                4, quad_scheme="default")
    Quad_vec = df.FunctionSpace(mesh, quad_vel)
    quad_fel = df.FiniteElement("Quadrature", mesh.ufl_cell(),
                                4, quad_scheme="default")
    Quad_scal = df.FunctionSpace(mesh, quad_fel)
    
    N = df.FacetNormal(mesh)

    ffun = df.MeshFunction("size_t", mesh, 1)
    ffun.set_all(2)
    dir_sub = df.CompiledSubDomain("near(x[1], 0)")
    dir_sub.mark(ffun, 0)
    
    neu_sub = df.CompiledSubDomain("near(x[0], 0) || near(x[0], 1) || near(x[1], 1)")
    neu_sub.mark(ffun, 1)


    f0 = df.interpolate(df.Expression(("0.0", "1.0"), element=quad_vel), Quad_vec)
    
    material = mat.NeoHookean(f0, act, matparams, active_model = active_model,
                              T_ref = 1.0, dev_iso_split=False)
    
    def make_dirichlet_bcs(W):
        bcs = [df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0)), ffun, 0)]
        return bcs




    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[T, 1]]}}
    
    df.parameters["adjoint"]["stop_annotating"] = True
    df.set_log_active(True)
    df.set_log_level(df.INFO)
    solver = LVSolver(params)
    solver.solve()

    uh,ph = solver.get_state().split(deepcopy=True)
    F = df.grad(uh)+df.Identity(2)
    J = df.project(df.det(F), DG1)

    err_u.append(df.errornorm(u_exact, uh, "H1", mesh = mesh))
    err_p.append(df.errornorm(p_exact, ph, "L2", mesh = mesh))
    err_J.append(df.errornorm(df.Expression("1.0"), J, "L2", mesh = mesh))

    if 0:
        u = df.interpolate(u_exact, P2)
        p = df.interpolate(p_exact, P1)

        df.plot(u-uh, title ="u-uh")
        df.plot(p-ph, title = "p-ph")
        df.plot(ph, title = "ph")
        df.plot(p, title = "p")
        df.plot(u, mode="displacement")
        df.plot(uh, interactive=True, mode="displacement")
        exit()

import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
h  = 1.0/np.array(ndivs)
plt.loglog(h, err_u, "b-o", label = r"$\|u - u_h\|_{H^1}$")
plt.loglog(h, err_p, "r-o", label = r"$\|p - p_h\|_{L^2}$")
plt.loglog(h, err_J, "g-o", label = r"$\|J - 1\|_{L^2}$")
plt.legend(loc = "best")
plt.show()
