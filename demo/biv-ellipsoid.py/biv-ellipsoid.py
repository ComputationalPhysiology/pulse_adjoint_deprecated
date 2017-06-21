import dolfin as df
import mshr, os

from pulse_adjoint.models import material  as mat
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.setup_parameters import setup_general_parameters
from pulse_adjoint.iterate import iterate

from pulse_adjoint.postprocess.utils import (smooth_from_points,
                                             localproject,
                                             remove_extreme_outliers)
setup_general_parameters()

base_x = 0.0

### LV
# The center of the LV ellipsoid
center_lv = df.Point(0.0, 0.0, 0.0)
a_lv_epi = 2.0
b_lv_epi = 1.0
c_lv_epi = 1.0

a_lv_endo = 1.5
b_lv_endo = 0.5
c_lv_endo = 0.5


### RV
# The center of the RV ellipsoid (slightl translated)
center_rv = df.Point(0.0, 0.5, 0.0)

a_rv_epi = 1.75
b_rv_epi = 1.5
c_rv_epi = 1.0

a_rv_endo = 1.45
b_rv_endo = 1.25
c_rv_endo = 0.75



## Markers
base_marker = 10
endolv_marker = 30
endorv_marker = 20
epi_marker = 40

class EndoLV(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0]-center_lv.x())**2/a_lv_endo**2 \
            + (x[1]-center_lv.y())**2/b_lv_endo**2 \
            + (x[2]-center_lv.z())**2/c_lv_endo**2 -1 < df.DOLFIN_EPS and on_boundary

class Base(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > base_x - df.DOLFIN_EPS and on_boundary

class EndoRV(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0]-center_rv.x())**2/a_rv_endo**2 \
            + (x[1]-center_rv.y())**2/b_rv_endo**2 \
            + (x[2]-center_rv.z())**2/c_rv_endo**2 - 1 < df.DOLFIN_EPS   \
            and (x[0]-center_lv.x())**2/a_lv_epi**2 \
            + (x[1]-center_lv.y())**2/b_lv_epi**2 \
            + (x[2]-center_lv.z())**2/c_lv_epi**2 - 0.9 > df.DOLFIN_EPS) and on_boundary

class Epi(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0]-center_rv.x())**2/a_rv_epi**2 \
            + (x[1]-center_rv.y())**2/b_rv_epi**2 \
            + (x[2]-center_rv.z())**2/c_rv_epi**2 - 0.9 > df.DOLFIN_EPS   \
            and (x[0]-center_lv.x())**2/a_lv_epi**2 \
            + (x[1]-center_lv.y())**2/b_lv_epi**2 \
            + (x[2]-center_lv.z())**2/c_lv_epi**2 - 0.9 > df.DOLFIN_EPS and on_boundary

def create_mesh():


    # The plane cutting the base
    diam    = 10.0
    box = mshr.Box(df.Point(base_x,-2,-2),df.Point(diam,diam,diam))
    # Generate mesh


    # LV epicardium
    el_lv = mshr.Ellipsoid(center_lv, a_lv_epi, b_lv_epi, c_lv_epi)
    # LV endocardium
    el_lv_endo = mshr.Ellipsoid(center_lv, a_lv_endo, b_lv_endo, c_lv_endo)

    # LV geometry (subtract the smallest ellipsoid)
    lv = el_lv - el_lv_endo


    # LV epicardium
    el_rv = mshr.Ellipsoid(center_rv, a_rv_epi, b_rv_epi, c_rv_epi)
    # LV endocardium
    el_rv_endo = mshr.Ellipsoid(center_rv, a_rv_endo, b_rv_endo, c_rv_endo)
    
    # RV geometry (subtract the smallest ellipsoid)
    rv = el_rv - el_rv_endo - el_lv

    # BiV geometry
    m = lv + rv - box

    # Some refinement level
    N = 13

    # Create mesh
    domain = mshr.generate_mesh(m, N)
    # df.plot(domain, interactive = True)

    return domain


def load_mesh():

    meshfile = "biv_mesh.xdmf"
    f= df.XDMFFile(df.mpi_comm_world(), meshfile)
    if os.path.isfile(meshfile):
        mesh= df.Mesh()
        f.read(mesh)
    
    else:
        mesh = create_mesh()
        f.write(mesh)

    return mesh

def mark_mesh(mesh):


    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    endolv = EndoLV()
    endolv.mark(ffun, endolv_marker)

    base = Base()
    base.mark(ffun, base_marker)

    endorv = EndoRV()
    endorv.mark(ffun, endorv_marker)

    epi = Epi()
    epi.mark(ffun, epi_marker)

    # df.plot(ffun, interactive=True)
    return ffun


def create_fiberfield(mesh, ffun):

    vel = df.VectorElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    VV = df.FunctionSpace(mesh, vel)

    fiberfile = "fiber.xml"
    if os.path.isfile(fiberfile):
        f0 = df.Function(VV, fiberfile)
    else:
        
        from mesh_generation import generate_fibers
        fields = generate_fibers(mesh, ffun=ffun)
        f0 = fields[0]
        
        from paraview import fiber_to_xdmf
        fiber_to_xdmf(f0, "fiber")
        
        f = df.File(fiberfile)
        f << f0

    return f0



def main():
    
    
    mesh = load_mesh()
    ffun = mark_mesh(mesh)
    # df.plot(ffun, interactive=True)

    f0 = create_fiberfield(mesh, ffun)

    
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = df.DirichletBC(V.sub(0), df.Constant(0.0), ffun, base_marker)
        return no_base_x_tran_bc 

   
    V = df.FunctionSpace(mesh, "R", 0)
    gamma = df.Function(V)

    material = mat.HolzapfelOgden(f0, gamma, active_model = "active_stress", T_ref=100.0)

    spring = df.Constant(1.0, name ="spring_constant")
    
    # Facet Normal
    N = df.FacetNormal(mesh)

    pressure_lv = df.Constant(0.0)
    pressure_rv = df.Constant(0.0)


    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure_lv, endolv_marker],
                              [pressure_rv, endorv_marker]],
                   "robin":[[spring, base_marker]]}}
    

    df.parameters["adjoint"]["stop_annotating"] = True
    solver = LVSolver(params)
    u,p = solver.get_state().split(deepcopy=True)
    U = df.Function(u.function_space(),
                    name ="displacement")
    f = df.XDMFFile(df.mpi_comm_world(), "displacement.xdmf")
    
    solver.solve()

    u,p = solver.get_state().split(deepcopy=True)
    U.assign(u)
    f.write(U)
 
    
    plv = 5.0
    prv = 3.0
    iterate("pressure", solver, (plv, prv), {"p_lv":pressure_lv, "p_rv":pressure_rv})

    u,p = solver.get_state().split(deepcopy=True)
    U.assign(u)
    f.write(U)
    
    

    iterate("gamma", solver, 1.0, gamma, max_nr_crash=100,max_iters=100)
    u,p = solver.get_state().split(deepcopy=True)
    U.assign(u)
    f.write(U)

if __name__ == "__main__":
    main()    
