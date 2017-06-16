from dolfin import *

from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.setup_parameters import setup_general_parameters
from pulse_adjoint import LVTestPatient
from pulse_adjoint.models.material import *
from pulse_adjoint.iterate import iterate


def demo_heart():

    setup_general_parameters()
    patient = LVTestPatient()
    
    mesh = patient.mesh
    ffun = patient.ffun
    N = FacetNormal(mesh)


    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        # no_base_x_tran_bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), patient.BASE)
        no_base_x_tran_bc = DirichletBC(V.sub(0), Constant(0.0), patient.markers["BASE"][0])
        return no_base_x_tran_bc


    # Fibers
    f0 = patient.fiber
    
    # Contraction parameter
    # gamma = Constant(0.0)
    gamma = Function(FunctionSpace(mesh, "R", 0))
    # Pressure
    pressure = Constant(0.0)

    # Spring
    spring = Constant(0.1)

    
    # Set up material model
    matparams = {"a":2.28, "a_f":1.685, 
                "b":9.726, "b_f":15.779}
    material = HolzapfelOgden(patient.fiber, gamma, matparams,
                              active_model = "active_strain", T_ref = 1.0)
    # material = HolzapfelOgden(f0, gamma, active_model = "active_stress")
    # material = NeoHookean(f0, gamma, active_model = "active_stress")

    # Create parameters for the solver
    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, patient.markers["ENDO"][0]]],
                   "robin":[[spring, patient.markers["BASE"][0]]]}}

    parameters["adjoint"]["stop_annotating"] = True

    # Initialize solver
    solver = LVSolver(params)

    # Solve for the initial state
    solver.solve()
    u,p = solver.get_state().split()
    # u = solver.get_state()#.split()
    plot(u, mode="displacement", title = "Initial solve")

    # Put on some pressure and solve
    iterate("pressure", solver, 1.0, {"p_lv":pressure})
        
    u,p = solver.get_state().split()
    
    plot(u, mode="displacement",
         title = "Soulution after pressure change")
    

    # Put on some active contraction and solve
    iterate("gamma", solver, 0.1, gamma)    
    u,p = solver.get_state().split()

    plot(u, mode="displacement",
         title = "Solution after initiation of active contraction")
    plot(p, title = "hydrostatic pressure")
    
    interactive()

def demo_cube():
    setup_general_parameters()
    mesh = UnitCubeMesh(3,3,3)

    # Make some simple boundary conditions
    class Right(SubDomain):
        def inside(self, x, on_boundary): 
            return x[0] > (1.0 - DOLFIN_EPS) and on_boundary
    class Left(SubDomain):
        def inside(self, x, on_boundary): 
            return x[0] < DOLFIN_EPS and on_boundary
    class TopBottom(SubDomain):
        def inside(self, x, on_boundary):
            return (x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS) and on_boundary

    
    # Mark boundaries
    ffun = MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    left = Left()
    left_marker = 1
    left.mark(ffun, left_marker)

    right = Right()
    right_marker = 2
    right.mark(ffun, right_marker)

    topbottom = TopBottom()
    topbottom_marker = 3
    topbottom.mark(ffun, topbottom_marker)

    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = DirichletBC(V.sub(0), 0, topbottom_marker)
        return no_base_x_tran_bc

    # Spring Constant for Robin Condition
    spring = Constant(0.1, name ="spring_constant")

    # Facet Normal
    N = FacetNormal(mesh)

    # Pressure
    pressure = Expression("t", t = 0.1)

    # Fibers
    V_f = QuadratureSpace(mesh, 4)
    # V_f = VectorFunctionSpace(mesh, "CG", 1)
    # Unit field in x-direction
    f0 = interpolate(Expression(("1.0", "0.0", "0.0")), V_f)

    # Contraction parameter
    gamma = Constant(0.1)
    
    # Set up material model
    material = HolzapfelOgden(f0, gamma, active_model = "active_stress")
    
    # Solver parameters
    solver_parameters = setup_solver_parameters()
    solver_parameters = {"snes_solver":{}}
    solver_parameters["nonlinear_solver"] = "snes"
    solver_parameters["snes_solver"]["method"] = "newtonls"
    solver_parameters["snes_solver"]["maximum_iterations"] = 8
    solver_parameters["snes_solver"]["absolute_tolerance"] = 1e-5
    solver_parameters["snes_solver"]["linear_solver"] = "lu"

    # Create parameters for the solver
    params= {"mesh": mesh,
            "facet_function": ffun,
              "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, left_marker]],
                   "robin":[[spring, right_marker]]},
             "solve":solver_parameters}

    solver = LVSolver(params)
    
    solver.solve()

    u,p = solver.get_state().split()
    plot(u, mode="displacement", title = "displacement")
    plot(p, title = "hydrostatic pressure")

    postprocess = solver.postprocess()

    fiber_stress = postprocess.fiber_stress()
    plot(fiber_stress, title = "fiber stress")

    
    interactive()

if __name__ == "__main__":
    # demo_cube()
    demo_heart()
