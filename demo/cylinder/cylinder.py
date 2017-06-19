import mshr, os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pulse_adjoint.models import material  as mat
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.setup_parameters import setup_general_parameters
from pulse_adjoint.iterate import iterate

from pulse_adjoint.postprocess.utils import (smooth_from_points,
                                             localproject,
                                             remove_extreme_outliers)

# from paraview import fun_to_xdmf

setup_general_parameters()

base_marker = 2
basebottom_marker = base_marker
basetop_marker = base_marker
endo_marker = 1
epi_marker = 3

r_inner = 1.0
t = 1.0
r_outer = r_inner + t


class Endo(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1]**2 + x[2]**2 - r_inner**2 < df.DOLFIN_EPS and on_boundary

class BaseBottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] + 1.0 < df.DOLFIN_EPS and on_boundary

class BaseTop(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] - 1.0 > -df.DOLFIN_EPS and on_boundary

class Epi(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1]**2 + x[2]**2 - (r_outer-0.1)**2 > df.DOLFIN_EPS and on_boundary

    
def create_mesh():

    
    domain1 = mshr.Cylinder(df.Point(np.array([1.0, 0.0, 0.0])),
                            df.Point(np.array([-1.0, 0.0, 0.0])), r_inner, r_inner, 300)


    domain2 = mshr.Cylinder(df.Point(np.array([1.0, 0.0, 0.0])),
                            df.Point(np.array([-1.0, 0.0, 0.0])), r_outer, r_outer, 300)

    domain = domain2 - domain1
    mesh= mshr.generate_mesh(domain, 10)
    df.plot(mesh, interactive=True)

    return mesh


def load_mesh():

    meshfile = "cylinder_mesh.xdmf"
    comm = df.mpi_comm_world()
    f = df.XDMFFile(comm, meshfile)
    if os.path.isfile(meshfile):
        mesh = df.Mesh(comm)
        f.read(mesh)
        
    else:
        mesh = create_mesh()
        # f = df.File(meshfile)
        f.write(mesh)

    return mesh

def mark_mesh(mesh):

    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    endo = Endo()
    endo.mark(ffun, endo_marker)

    basetop = BaseTop()
    basetop.mark(ffun, basetop_marker)

    basebottom = BaseBottom()
    basebottom.mark(ffun, basebottom_marker)

    epi = Epi()
    epi.mark(ffun, epi_marker)
    # df.plot(ffun, interactive=True)
    

    return ffun

def create_fiberfield(mesh):

    vel = df.VectorElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    VV = df.FunctionSpace(mesh, vel)

    fel = df.FiniteElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    V = df.FunctionSpace(mesh, fel)
    
    fiberfile = "circ_fiber.xml"
    if os.path.isfile(fiberfile):
        f0 = df.Function(VV, fiberfile)
    else:
        from mesh_generation.strain_regions import make_unit_vector, fill_coordinates_ec
        dofs_x = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
        f0 = make_unit_vector(V, VV, dofs_x, fill_coordinates_ec)
        
        # fun_to_xdmf(f0, "fiber")
        f = df.File(fiberfile)
        f << f0

    return f0


def radial_vectorfield(mesh):

    def rad(i, e_c_x, e_c_y, e_c_z, coord, foci):
        norm = df.sqrt(coord[1]**2 + coord[2]**2)
        if not df.near(norm, 0):
            e_c_y.vector()[i] = coord[1]/norm
            e_c_z.vector()[i] = coord[2]/norm
        else:
            #We are at the apex where clr system doesn't make sense
            #So just pick something.
            e_c_y.vector()[i] = 1
            e_c_z.vector()[i] = 0


    vel = df.VectorElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    VV = df.FunctionSpace(mesh, vel)

    fel = df.FiniteElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    V = df.FunctionSpace(mesh, fel)

    fiberfile = "radial.xml"
    if os.path.isfile(fiberfile):
        f0 = df.Function(VV, fiberfile)
    else:
    
        from mesh_generation.strain_regions import make_unit_vector, fill_coordinates_ec
        dofs_x = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
        f0 = make_unit_vector(V, VV, dofs_x, rad)
        
        # fun_to_xdmf(f0, "radial")
        f = df.File(fiberfile)
        f << f0


    return f0

def longitudinal_vectorfield(mesh):


    def longitudinal(i, e_c_x, e_c_y, e_c_z, coord, foci):
        e_c_x.vector()[i] = -1.0
        
    vel = df.VectorElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    VV = df.FunctionSpace(mesh, vel)

    fel = df.FiniteElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    V = df.FunctionSpace(mesh, fel)

    fiberfile = "longitudinal.xml"
    if os.path.isfile(fiberfile):
        f0 = df.Function(VV, fiberfile)
    else:
    
        from mesh_generation.strain_regions import make_unit_vector
        dofs_x = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
        f0 = make_unit_vector(V, VV, dofs_x, longitudinal)
        
        # fun_to_xdmf(f0, "longitudinal")
        f = df.File(fiberfile)
        f << f0


    return f0

def radial_average(solver, r0, rs):

    Tf_ufl = solver.postprocess().cauchy_stress_component(r0, deviatoric=False)
    mesh=solver.parameters["mesh"]
    fel = df.FiniteElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    VV = df.FunctionSpace(mesh, fel)
    Tf = df.project(Tf_ufl, VV)
    V = df.FunctionSpace(mesh, "DG", 1)
    Tf_dg = smooth_from_points(V, Tf)

    if not os.path.exists("bmesh"):
        os.makedirs("bmesh")

    avg_stress = []
    for r in rs:

        mshfile = "/".join(["bmesh", "cylinder_{}.xdmf".format(r)])
        f = df.XDMFFile(df.mpi_comm_world(), mshfile)
        if not os.path.exists(mshfile):
            
            domain = mshr.Cylinder(df.Point(np.array([1.0, 0.0, 0.0])),
                                   df.Point(np.array([-1.0, 0.0, 0.0])), r,r, 300)

            cylmesh= mshr.generate_mesh(domain, 10)
            f.write(cylmesh)
        else:
            cylmesh = df.Mesh(df.mpi_comm_world())
            f.read(cylmesh)
            

        V0 = df.FunctionSpace(cylmesh, "DG", 1)
        v = df.interpolate(Tf_dg, V0)
    
        rdom = df.AutoSubDomain(lambda x, on_bnd: (x[1]**2 + x[2]**2 - (0.1)**2 > -df.DOLFIN_EPS) \
                                and on_bnd and x[0] < 1.0 - df.DOLFIN_EPS and x[0] > -1.0 + df.DOLFIN_EPS)
    
        ffun = df.FacetFunction("size_t", cylmesh)
        ffun.set_all(0)
        rdom.mark(ffun, 1)
        ds = df.Measure("ds", domain=cylmesh, subdomain_data=ffun)(1)


        area =  df.assemble(df.Constant(1.0)*ds)
        Tf_r =  df.assemble(v*ds) / area
        avg_stress.append(Tf_r)

    return avg_stress
   
def compute_stress(solver, r0, h5name):

    
    Tf_ufl = solver.postprocess().cauchy_stress_component(r0, deviatoric=False)
    Ef_ufl = solver.postprocess().green_strain_component(r0)

    mesh=solver.parameters["mesh"]
    fel = df.FiniteElement("Quadrature", mesh.ufl_cell(), 4, quad_scheme="default")
    VV = df.FunctionSpace(mesh, fel)

    Tf = df.project(Tf_ufl, VV)
    # fun_to_xdmf(Tf, "{}_stress_quad".format(h5name))
    Ef = df.project(Ef_ufl, VV)
    # fun_to_xdmf(Ef, "{}_strain_quad".format(h5name))

    V = df.FunctionSpace(mesh, "DG", 1)
    Tf_dg = smooth_from_points(V, Tf)
    Ef_dg = smooth_from_points(V, Ef)
    
    f = df.XDMFFile(df.mpi_comm_world(), "{}_stress_dg.xdmf".format(h5name))
    f.write(Tf_dg)

    
    f = df.XDMFFile(df.mpi_comm_world(), "{}_strain_dg.xdmf".format(h5name))
    f.write(Ef_dg)
    
    

def main():


    mesh = load_mesh()
    ffun = mark_mesh(mesh)
    

    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = df.DirichletBC(V.sub(0), df.Constant(0.0), ffun, base_marker)
        return no_base_x_tran_bc

    f0 = create_fiberfield(mesh)
    r0 = radial_vectorfield(mesh)
    l0 = longitudinal_vectorfield(mesh)


    V = df.FunctionSpace(mesh, "R", 0)
    gamma = df.Function(V)

    matparams = {"mu":15.0}
    
    # Set up material model
    # material = mat.HolzapfelOgden(f0, gamma, active_model = "active_strain", T_ref=0.25)
    material = mat.HolzapfelOgden(f0, gamma, active_model = "active_stress", T_ref=30.0)
    # material = mat.NeoHookean(f0, gamma, matparams, active_model = "active_stress", T_ref=30.0)
    # material = mat.NeoHookean(f0, gamma, matparams, active_model = "active_strain", T_ref=0.3)
    # material = mat.LinearElastic(f0, gamma, active_model = "active_stress", T_ref=30.0)
    # material = mat.Guccione(f0, gamma, active_model = "active_stress", T_ref=30.0, s0 = r0, n0 = l0)
    
    # Spring
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

    solver = LVSolver(params)
    solver.solve()


    mat_str = "_".join([material.get_material_model(), material.get_active_model(), "dev"])
    if not os.path.exists(mat_str):
        os.makedirs(mat_str)

    iterate("pressure", solver, 10.0, {"p_lv":pressure})

    Tf = compute_stress(solver, r0, "{}/radial_passive".format(mat_str))
    Tf = compute_stress(solver, f0, "{}/circ_passive".format(mat_str))

    
    u,p = solver.get_state().split(deepcopy=True)
    f = df.XDMFFile(df.mpi_comm_world(), "{}/disp_passive.xdmf".format(mat_str))
    f.write(u)
    
    # df.plot(u, interactive=True, mode="displacement")
    iterate("gamma", solver, 1.0, gamma, max_nr_crash=100,max_iters=100)

    Tf = compute_stress(solver, r0, "{}/radial_acitve".format(mat_str))
    Tf = compute_stress(solver, f0, "{}/circ_active".format(mat_str))
    
    u,p = solver.get_state().split(deepcopy=True)
    f = df.XDMFFile(df.mpi_comm_world(), "{}/disp_active.xdmf".format(mat_str))
    f.write(u)
    
    # df.plot(u, interactive=True, mode="displacement")
    # df.plot(ffun,interactive=True)


def compare_analytic_passive():

    mesh = load_mesh()
    ffun = mark_mesh(mesh)
    

    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = df.DirichletBC(V.sub(0), df.Constant(0.0), ffun, base_marker)
        return no_base_x_tran_bc

    f0 = create_fiberfield(mesh)
    r0 = radial_vectorfield(mesh)
    l0 = longitudinal_vectorfield(mesh)


    V = df.FunctionSpace(mesh, "R", 0)
    gamma = df.Function(V)

    matparams = {"mu":15.0}

    spring = df.Constant(1.0, name ="spring_constant")
    # Facet Normal
    N = df.FacetNormal(mesh)


    P = 10.0
    colors = sns.color_palette("hls", 5)

    fig = plt.figure()
    ax = fig.gca()
    
    for i, Mat in enumerate([mat.HolzapfelOgden, mat.NeoHookean, mat.LinearElastic, mat.Guccione]):

        
        material = Mat(f0, gamma, active_model = "active_stress", T_ref=30.0, s0=r0,n0=l0)
        matstr=material.get_material_model()
        print matstr

        pressure = df.Constant(0.0)
        
        params= {"mesh": mesh,
                 "facet_function": ffun,
                 "facet_normal": N,
                 "material": material,
                 "bc":{"dirichlet": make_dirichlet_bcs,
                       "neumann":[[pressure, endo_marker]],
                       "robin":[[spring, base_marker]]}}

        solver = LVSolver(params)
        solver.solve()

        
        
        iterate("pressure", solver, P, {"p_lv":pressure})

        rs = np.linspace(1.0, 2.0, 10)
        avg_circ_stress = radial_average(solver, f0, rs)
        avg_rad_stress = radial_average(solver, r0, rs)
        print avg_circ_stress
        print avg_rad_stress
        ax.plot(rs, avg_circ_stress, label="circ stress {}".format(matstr),
                color = colors[i], linestyle = "-")
        ax.plot(rs, avg_rad_stress, label="radial stress {}".format(matstr),
                color = colors[i], linestyle = "-.")
        

    
    from analytic_stress import circ_stress, rad_stress
    c_stress = circ_stress(P, rs)
    r_stress = rad_stress(P, rs)
    ax.plot(rs, c_stress, label="analtic circ stress",  color = colors[-1], linestyle = "-")
    ax.plot(rs, r_stress, label="analytic radial stress",  color = colors[-1], linestyle = "-.")
    # ax.legend(loc = "best")
    ax.set_ylabel("Average Stress (kPa)")
    ax.set_title("Cylinder radius 1-2, p = 10.0")
    ax.set_xlabel("Radius")
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5)) 
    fig.savefig("passive_stress_cylinder.png",
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

def compare_analytic_active():

    mesh = load_mesh()
    ffun = mark_mesh(mesh)
    

    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = df.DirichletBC(V.sub(0), df.Constant(0.0), ffun, base_marker)
        return no_base_x_tran_bc

    f0 = create_fiberfield(mesh)
    r0 = radial_vectorfield(mesh)
    l0 = longitudinal_vectorfield(mesh)


    
    matparams = {"mu":15.0}

    spring = df.Constant(1.0, name ="spring_constant")
    # Facet Normal
    N = df.FacetNormal(mesh)

    # active_model = "active_stress"
    active_model = "active_strain"

    if active_model == "active_stress":
        T_ref = 10.0
    else:
        T_ref = 0.1

    
    colors = sns.color_palette("hls", 5)

    P = 10.0
    fig = plt.figure()
    ax = fig.gca()
    
    for i, Mat in enumerate([mat.HolzapfelOgden, mat.NeoHookean, mat.LinearElastic, mat.Guccione]):

        V = df.FunctionSpace(mesh, "R", 0)
        gamma = df.Function(V)
        
        material = Mat(f0, gamma, active_model = active_model, T_ref=T_ref, s0=r0,n0=l0)
        matstr=material.get_material_model()
        print matstr
        
        pressure = df.Constant(0.0)
       

        
        params= {"mesh": mesh,
                 "facet_function": ffun,
                 "facet_normal": N,
                 "material": material,
                 "bc":{"dirichlet": make_dirichlet_bcs,
                       "neumann":[[pressure, endo_marker]],
                       "robin":[[spring, base_marker]]}}

        solver = LVSolver(params)
        solver.solve()

        
        
        iterate("pressure", solver, P, {"p_lv":pressure})
        iterate("gamma", solver, 1.0, gamma)

        rs = np.linspace(1.0, 2.0, 10)
        avg_circ_stress = radial_average(solver, f0, rs)
        avg_rad_stress = radial_average(solver, r0, rs)
        ax.plot(rs, avg_circ_stress, label="circ stress {}".format(matstr),
                color = colors[i], linestyle = "-")
        ax.plot(rs, avg_rad_stress, label="radial stress {}".format(matstr),
                color = colors[i], linestyle = "-.")
        

    
    from analytic_stress import circ_stress, rad_stress
    c_stress = circ_stress(P, rs)
    r_stress = rad_stress(P, rs)
    ax.plot(rs, c_stress, label="analtic circ stress",  color = colors[-1], linestyle = "-")
    ax.plot(rs, r_stress, label="analytic radial stress",  color = colors[-1], linestyle = "-.")
    # ax.legend(loc = "best")
    ax.set_ylabel("Average Stress (kPa)")
    ax.set_title("Cylinder radius 1-2, p = 10.0, {}, Tref = {}".format(active_model, T_ref))
    ax.set_xlabel("Radius")
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5)) 
    fig.savefig("active_stress_cylinder_{}.png".format(active_model),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()
    
    

if __name__ == "__main__":
    # main()
    # compare_analytic_passive()
    compare_analytic_active()
