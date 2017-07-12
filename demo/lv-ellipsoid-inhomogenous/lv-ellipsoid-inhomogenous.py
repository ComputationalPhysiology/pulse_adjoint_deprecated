import numpy as np

import dolfin as df
import mshr, os

from pulse_adjoint.models import material  as mat
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.setup_parameters import setup_general_parameters
from pulse_adjoint.iterate import iterate
from pulse_adjoint.setup_optimization import RegionalParameter

from mesh_generation.idealized_geometry import mark_strain_regions


from pulse_adjoint.postprocess.utils import (smooth_from_points,
                                             localproject,
                                             remove_extreme_outliers)

setup_general_parameters()

base_x = 0.0

### LV
# The center of the LV ellipsoid
center = df.Point(0.0, 0.0, 0.0)
a_epi = 2.0
b_epi = 1.0
c_epi = 1.0

a_endo = 1.5
b_endo = 0.5
c_endo = 0.5


## Markers
base_marker = 10
endo_marker = 30

epi_marker = 40

class Endo(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0]-center.x())**2/a_endo**2 \
            + (x[1]-center.y())**2/b_endo**2 \
            + (x[2]-center.z())**2/c_endo**2 -1.1 < df.DOLFIN_EPS and on_boundary

class Base(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] - base_x < df.DOLFIN_EPS and on_boundary

class Epi(df.SubDomain):
    def inside(self, x, on_boundary):
        return  (x[0]-center.x())**2/a_epi**2 \
            + (x[1]-center.y())**2/b_epi**2 \
            + (x[2]-center.z())**2/c_epi**2 - 0.9 > df.DOLFIN_EPS and on_boundary

def create_mesh():


    # The plane cutting the base
    diam    = -10.0
    box = mshr.Box(df.Point(base_x,2,2),df.Point(diam,diam,diam))
    # Generate mesh


    # LV epicardium
    el_lv = mshr.Ellipsoid(center, a_epi, b_epi, c_epi)
    # LV endocardium
    el_lv_endo = mshr.Ellipsoid(center, a_endo, b_endo, c_endo)

    # LV geometry (subtract the smallest ellipsoid)
    lv = el_lv - el_lv_endo

    
    # LV geometry
    m = lv-box

    # Some refinement level
    N = 13

    # Create mesh
    domain = mshr.generate_mesh(m, N)
    # df.plot(domain, interactive = True)
    # exit()

    return domain


def load_mesh():

    meshfile = "lv_mesh.xdmf"
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

    endo = Endo()
    endo.mark(ffun, endo_marker)

    base = Base()
    base.mark(ffun, base_marker)

  
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
    # df.plot(mesh, interactive=True)
    # exit()
    ffun = mark_mesh(mesh)
    # df.plot(ffun, interactive=True)
    # exit()
    sfun = mark_strain_regions(mesh, -1.0, [6,6,4,1])


    f0 = create_fiberfield(mesh, ffun)

    
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = df.DirichletBC(V.sub(0), df.Constant(0.0), ffun, base_marker)
        return no_base_x_tran_bc 

   
    # V = df.FunctionSpace(mesh, "R", 0)
    # gamma = df.Function(V)
    gamma = RegionalParameter(sfun)

    gamma_base = [1,1,1,1,0.5,0.6]
    gamma_mid =  [1,1,1,1,0.5,0.6]
    gamma_apical =  [1,1,0.55,0.6]
    gamma_apex = [0.6]
    

    gamma_arr = np.array(gamma_base + gamma_mid + gamma_apical + gamma_apex)


    G = RegionalParameter(sfun)
    G.vector()[:] = gamma_arr
    G_ = df.project(G.get_function(), G.get_ind_space())
    # G = gamma.g
    f_gamma = df.XDMFFile(df.mpi_comm_world(), "activation.xdmf")
    f_gamma.write(G_)
    

    material = mat.HolzapfelOgden(f0, gamma, active_model = "active_strain", T_ref=0.25)

    spring = df.Constant(1.0, name ="spring_constant")
    
    # Facet Normal
    N = df.FacetNormal(mesh)

    pressure = df.Constant(0.0)



    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, endo_marker]],
                   "robin":[[spring, base_marker]]}}
    

    df.parameters["adjoint"]["stop_annotating"] = True
    solver = LVSolver(params)
    u,p = solver.get_state().split(deepcopy=True)
    U = df.Function(u.function_space(),
                    name ="displacement")
    f = df.XDMFFile(df.mpi_comm_world(), "displacement.xdmf")
    
    solver.parameters["solve"]["newton_solver"]["report"] =True
    df.set_log_active(True)
    df.set_log_level(20)
    # from IPython import embed; embed()
    # exit()
    
    solver.solve()

    u,p = solver.get_state().split(deepcopy=True)
    U.assign(u)
    f.write(U)
 
    
    plv = 5.0
    iterate("pressure", solver, plv, {"p_lv":pressure})

    u,p = solver.get_state().split(deepcopy=True)
    U.assign(u)
    f.write(U)
    
    
    g = 1.0
    g_ = g * gamma_arr
    
    iterate("gamma", solver, g_, gamma, max_nr_crash=100,max_iters=100)
    u,p = solver.get_state().split(deepcopy=True)
    U.assign(u)
    f.write(U)

if __name__ == "__main__":
    main()    
