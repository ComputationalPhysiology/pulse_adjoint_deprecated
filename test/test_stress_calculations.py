"""

Test that stresses are computed correctly for the two active models 
(active strain  and active stress) and the three material models 
(neo_hookean, holzapfel_ogden and guccione) with and without splitting
the deformation gradient into isochoring and deviatoric component. 

"""


from dolfin import *
import numpy as np
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.models.material import HolzapfelOgden, NeoHookean, Guccione
from pulse_adjoint.setup_optimization import (setup_solver_parameters,
                                              setup_general_parameters)
from pulse_adjoint.adjoint_contraction_args import logger
from pulse_adjoint.iterate import iterate_gamma, iterate
from pulse_adjoint.utils import QuadratureSpace

dev_iso_splits = [True, False]
material_models = ["neo_hookean", "holzapfel_ogden", "guccione"]

N = 3
mesh = UnitCubeMesh(N,N,N)

# Make some simple boundary conditions
class Free(SubDomain):
    def inside(self, x, on_boundary): 
        return x[0] > (1.0 - DOLFIN_EPS) and on_boundary
class Fixed(SubDomain):
    def inside(self, x, on_boundary): 
        return x[0] < DOLFIN_EPS and on_boundary
 
    
# Mark boundaries
ffun = MeshFunction("size_t", mesh, 2)
ffun.set_all(0)

fixed = Fixed()
fixed_marker = 1
fixed.mark(ffun, fixed_marker)

free = Free()
free_marker = 2
free.mark(ffun, free_marker)


# Facet Normal
N = FacetNormal(mesh)

# Fibers
V_f = QuadratureSpace(mesh, 4)

f0 = interpolate(Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
s0 = interpolate(Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
n0 = interpolate(Expression(("0.0", "0.0", "1.0"), degree=1), V_f)


def test_active_stress(dev_iso_split=True, material_model ="holzapfel_ogden"):
    """
    Fix x  = 0 plane and apply a force in the x-direction of 20 kPa.
    Apply an equal active stress. Then the displacement should be 
    zero and the total fiber stress should equal the force applied.
    """

    

    active_value = 20.0
    
    # Pressure
    pressure = Constant(-active_value)
    # pressure = Constant(0.0)

    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), fixed)
        return no_base_x_tran_bc

    # Contraction parameter
    # V = FunctionSpace(mesh, "R", 0)
    # gamma = Function(V)
    gamma = Constant(1.0)
    T_ref = active_value
 
    
    # Set up material model
    if material_model == "guccione":
        Material = Guccione
    elif material_model == "neo_hookean":
        Material = NeoHookean
    else:
        Material = HolzapfelOgden

    matparams = Material.default_parameters()
    
    material = Material(f0, gamma, matparams,
                        active_model = "active_stress",
                        T_ref =T_ref, s0 = s0, n0 = n0,
                        dev_iso_split = dev_iso_split)


    solver_parameters = setup_solver_parameters()
    # solver_parameters["snes_solver"]["report"] = True
    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, free_marker]]},
             "solve":solver_parameters}

    solver = LVSolver(params)
    solver.solve()

    # iterate("pressure", solver, active_value, {"p_lv":pressure})
    # iterate("gamma", solver, 1.0, gamma)
    u,p = solver.get_state().split(deepcopy = True)

    
    F = solver._F
    dim = 3
    f = F*f0

    I = Identity(dim)
    J = det(F)

    T = material.CauchyStress(F, p)
    V_dg = FunctionSpace(mesh, "DG", 1)

    
    Tf = inner(T*f/f**2, f)
    Tf_dg = project(Tf, V_dg)

    tol = 1e-10

    

    # plot(Tf_dg, title="Tf_df")
    # plot(p, title ="hydrostatic pressure")
    # plot(u,mode="displacement", interactive=True)
    # exit()

    
    assert all(abs(u.vector().array()) < tol)

    if not dev_iso_split:
        
        if material_model == "guccione":
            assert all(abs(p.vector().array()) < tol)
            assert all(abs(Tf_dg.vector().array()) < tol)
            
        elif material_model == "holzapfel_ogden":
            assert all(abs(p.vector().array() - matparams["a"]) < tol)
            assert all(abs(Tf_dg.vector().array()  + matparams["a"]) < tol)
        else:
            assert all(abs(p.vector().array() - matparams["mu"]) < tol)
            assert all(abs(Tf_dg.vector().array() + matparams["mu"]) < tol)

    else:
        assert all(abs(p.vector().array()) < tol)
        assert all(abs(Tf_dg.vector().array()) < tol)




def test_active_strain(dev_iso_split=False, material_model ="guccione"):
    """
    Test that when the cube contracts in the x - direction and 
    only homogeneous direclet in the x - direction at x = 0, 
    then the stresses should be zero
    """

    
    # Pressure
    pressure = Expression("-t", t = 0, degree=1)



    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = DirichletBC(V.sub(0), Constant(0.0), fixed, "pointwise")
        return no_base_x_tran_bc


    # Contraction parameter
    gamma = Constant(0.3)
    T_ref = 1.0
 
    
    # Set up material model
    if material_model == "guccione":
        Material = Guccione
    elif material_model == "neo_hookean":
        Material = NeoHookean
    else:
        Material = HolzapfelOgden

   
    matparams = Material.default_parameters()
    
    material = Material(f0, gamma, matparams,
                        active_model = "active_strain",
                        T_ref =T_ref, s0 = s0, n0 = n0,
                        dev_iso_split = dev_iso_split)


    solver_parameters = setup_solver_parameters()
    # solver_parameters["snes_solver"]["report"] = True
    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, free_marker]]},
             "solve":solver_parameters}

    solver = LVSolver(params)
    solver.solve()
    u,p = solver.get_state().split(deepcopy = True)

    logger
    F = solver._F
    dim = 3
    f = F*f0

    I = Identity(dim)
    J = det(F)

    
    T = material.CauchyStress(F, p)
    
    V_dg = FunctionSpace(mesh, "DG", 1)

    
    Tf = inner(T*f/f**2, f)
    Tf_dg = project(Tf, V_dg)

     
   

    #We have to be kind with the tolerance here
    tol = 1e-4
    

    
    if not dev_iso_split:
        
        
        if material_model == "guccione":
            assert all(abs(p.vector().array()) < tol)
            assert all(abs(Tf_dg.vector().array()) < tol)
        elif material_model == "holzapfel_ogden":
            assert all(abs(p.vector().array() - matparams["a"]) < tol)
            assert all(abs(Tf_dg.vector().array() + matparams["a"]) < tol)
        else:
            assert all(abs(p.vector().array() - matparams["mu"]) < tol)
            assert all(abs(Tf_dg.vector().array() + matparams["mu"]) < tol)

    else:

        assert all(abs(p.vector().array()) < tol)
        assert all(abs(Tf_dg.vector().array()) < tol)
       

    # plot(Tf_dg, title="Tf_df")
    # plot(p, title ="hydrostatic pressure")
    # plot(u,mode="displacement", interactive=True)
        
    
def test_all():


    # Active strain
    print "active strain"
    for dev_iso_split in dev_iso_splits:
        print dev_iso_split
        for material_model in material_models:
            print material_model
            test_active_strain(dev_iso_split, material_model)
            

    # Active stress
    print "active stress"
    for dev_iso_split in dev_iso_splits:
        print dev_iso_split
        for material_model in material_models:
            print material_model
            test_active_stress(dev_iso_split, material_model)
            

    
    

if __name__ == "__main__":
    setup_general_parameters()
    test_all()
