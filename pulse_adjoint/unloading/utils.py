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
__author__ = "Henrik Finsberg (henriknf@simula.no)"
import os
import dolfin as df
from ..numpy_mpi import *
from ..adjoint_contraction_args import logger

try:
    import h5py
    has_h5py=True
except:
    has_h5py=False

class ResidualCalculator(object):
    def __init__(self, mesh):
	self.mesh = mesh
	d = self.mesh.topology().dim()
	self.bbtree = df.BoundingBoxTree()
	local_points = [v.point() for v in df.vertices(self.mesh)]
	coords = [(p.x(), p.y(), p.z()) for p in local_points]
	
	coords = gather_broadcast(np.array(coords).flatten())
	coords.resize(len(coords)/d, d)
	glob_points = [df.Point(p) for p in coords]
	self.bbtree.build(glob_points, 3)		
        
    def calculate_residual(self, mesh2):
	boundmesh = df.BoundaryMesh(mesh2, "exterior")
	d = max([self.bbtree.compute_closest_point(df.Vertex(boundmesh, v_idx).point())[1] for v_idx in xrange(boundmesh.num_vertices())])
	return df.MPI.max(df.mpi_comm_world(), d)

class Object(object):pass

def save(obj, h5name, name, h5group = ""):
    """
    Save object to and HDF file. 
    
    Parameters
    ----------
    
    obj : dolfin.Mesh or dolfin.Function
        The object you want to save
    name : str
        Name of the object
    h5group : str
        The folder you want to save the object 
        withing the HDF file. Default: ''

    """
    
    group = "/".join([h5group, name])
    file_mode = "a" if os.path.isfile(h5name) else "w"

    if os.path.isfile(h5name):
        from ..io.utils import check_and_delete
        check_and_delete(h5name, group)
        file_mode = "a"
    else:
        file_mode = "w"

    logger.debug("Save {0} to {1}:{2}/{0}".format(name,
                                                 h5name,
                                                 h5group))

    if isinstance(obj,df.Function) and \
       obj.ufl_element().family() == "Quadrature":
        
        quad_to_xdmf(obj, h5name, group, file_mode)

    else:
        with df.HDF5File(df.mpi_comm_world(), h5name, file_mode) as h5file:
            h5file.write(obj, group)

def quad_to_xdmf(obj, h5name, h5group = "", file_mode = "w"):

    V = obj.function_space()
    gx, gy, gz = obj.split(deepcopy=True)
            
    W = V.sub(0).collapse()
    coords_tmp = gather_broadcast(W.tabulate_dof_coordinates())
    coords = coords_tmp.reshape((-1, 3))
    u = gather_broadcast(gx.vector().array())
    v = gather_broadcast(gy.vector().array())
    w = gather_broadcast(gz.vector().array())
    vecs = np.array([u,v,w]).T
    from ..io.utils import open_h5py, parallel_h5py
    with open_h5py(h5name) as h5file:

        if not parallel_h5py:
            if df.mpi_comm_world().rank == 0:
            
                h5file.create_dataset("/".join([h5group, "coordinates"]), data=coords)
                h5file.create_dataset("/".join([h5group, "vector"]), data=vecs)
        else:
            h5file.create_dataset("/".join([h5group, "coordinates"]), data=coords)
            h5file.create_dataset("/".join([h5group, "vector"]), data=vecs)    

            

def inflate_to_pressure(pressure, solver, p_expr, is_biv = None, ntries = 5, n = 2,
                        annotate = False):

    if is_biv == None: is_biv = isinstance(pressure, tuple) and len(pressure) == 2
    solve = solve_biv if is_biv else solve_lv

    logger.debug("\nInflate geometry to p = {} kPa".format(pressure))
    w = solve(pressure, solver,p_expr, ntries, n, annotate)
    
    return solver.get_displacement(annotate = annotate)

def print_volumes(geometry, logger=logger, is_biv = False):
    
    logger.info(("\nLV Volume of original geometry = "\
                 "{:.3f} ml".format(get_volume(geometry))))
    if is_biv:
        logger.info(("RV Volume of original geometry = "\
                     "{:.3f} ml".format(get_volume(geometry, chamber="rv"))))
        
def update_geometry(geometry, u= None, regen_fibers = True, *args):

    new_mesh = df.Mesh(geometry.mesh)

    if u is not None:
        U = move(new_mesh, u, -1.0, *args)
    else:
        U = u
    
    new_geometry = copy_geometry(new_mesh, geometry)

    fields = ['fiber', 'sheet', 'sheet_normal']
    local_basis = ['circumferential','radial', 'longitudinal']

    for attr in fields + local_basis:
        if hasattr(geometry, attr) and getattr(geometry, attr) is not None:

            regen = regen_fibers if attr == "fiber" else False
            f0 = getattr(geometry, attr).copy()
           
            f = update_vector_field(f0,
                                    new_mesh, U,
                                    regen_fibers = regen)
            setattr(new_geometry, attr, f)
    
    return new_geometry

def copy_geometry(new_mesh, geometry):

    new_geometry = Object()
    new_geometry.mesh = new_mesh
    new_geometry.markers = geometry.markers
    new_geometry.ffun = df.MeshFunction("size_t", new_mesh, 2, new_mesh.domains())
    new_geometry.ffun.set_values(geometry.ffun.array())
    new_geometry.sfun = df.MeshFunction("size_t", new_mesh, 3, new_mesh.domains())
    new_geometry.sfun.set_values(geometry.sfun.array())
    return new_geometry

def move(mesh, u, factor= 1.0, approx = "project"):

    W = df.VectorFunctionSpace(u.function_space().mesh(), "CG", 1)

    # msg = "CG1 approximation can be either 'project' or 'interpolate'"
    # assert approx in ["project", "interpolate"], msg
    
    # if approx == "interpolate":
    #     # Ideally we would do this 
    #     u_int = interpolate(u, W)
    # else:
    #     # This only works with dolfin-adjoint
    #     u_int = project(u, W)

    # Use interpolation for now. It is the only thing that makes sense
    u_int = df.interpolate(u, W)

        
    u0 = df.Function(W)
    arr = factor*gather_broadcast(u_int.vector().array())
    assign_to_vector(u0.vector(), arr)
    
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    U = df.Function(V)
    assign_to_vector(U.vector(), arr)
    
    df.ALE.move(mesh, U)

    return u0



    

def setup_general_parameters():
    """
    Parameters to speed up the compiler
    """
    from ..setup_parameters import setup_general_parameters
    setup_general_parameters()
    

def list_sum(l):
    """
    Return the sum of a list, when the convetiional
    method (like `sum`) it not working.
    For example if you have a list of dolfin functions.

    :param list l: a list of objects 
    :returns: The sum of the list. The type depends on 
              the type of elemets in the list

    """
    
    if not isinstance(l, list):
        return l

    out = l[0]
    for item in l[1:]:
        out += item
    return out



def solve_biv(pressure, solver, p_expr, ntries = 5, n = 2, annotate = False):


        
    df.parameters["adjoint"]["stop_annotating"] = True
    from ..iterate import iterate, logger as logger_it
    ps, states = iterate("pressure", solver, pressure, p_expr)

    level = logger_it.level
    logger_it.setLevel(df.WARNING)
    
    if annotate:
        # Only record the last solve, otherwise it becomes too
        # expensive on the memory. 
        df.parameters["adjoint"]["stop_annotating"] = not annotate
        solver.solve()

    logger_it.setLevel(level)
    w = solver.get_state().copy(True)
    return w



def solve_lv(pressure, solver, p_expr, ntries = 5, n = 2, annotate = False):

    df.parameters["adjoint"]["stop_annotating"] = True
    from ..iterate import iterate, logger as logger_it
    
    level = logger_it.level
    logger_it.setLevel(df.WARNING)
    
    ps, states = iterate("pressure", solver, pressure, p_expr)
    if annotate:
        # Only record the last solve, otherwise it becomes too
        # expensive on the memory. 
        df.parameters["adjoint"]["stop_annotating"] = not annotate
        solver.solve()

    logger_it.setLevel(level)
    w = solver.get_state().copy(True)
    return w

    
    


def update_vector_field(f0, new_mesh, u = None, name = "fiber", normalize =True, regen_fibers = False):
    
    ufl_elem = f0.function_space().ufl_element()
    f0_new = df.Function(df.FunctionSpace(new_mesh,ufl_elem), name = name)

    mpi_size = df.mpi_comm_world().getSize()

    
    if regen_fibers and mpi_size == 1:

        from mesh_generation.generate_mesh import setup_fiber_parameters
        from mesh_generation.mesh_utils import load_geometry_from_h5, generate_fibers
        fiber_params = setup_fiber_parameters()
        fields = generate_fibers(new_mesh, fiber_params)
        f0_new = fields[0]

    else:

        if regen_fibers:
            msg = ("Warning fibers can only be regenerated in serial. "+
                   "Use Piola transformation instead.\n")
            logger.warning(msg)
        
        if u is not None:
            
            f0_mesh = f0.function_space().mesh()
            u_elm = u.function_space().ufl_element()
            V = df.FunctionSpace(f0_mesh, u_elm)
            u0 = df.Function(V)
            arr = gather_broadcast(u.vector().array())
            assign_to_vector(u0.vector(), arr)
            
            F = df.grad(u0) + df.Identity(3)
     
            f0_updated = df.project(F*f0, f0.function_space())


            if normalize:
                f0_updated = normalize_vector_field(f0_updated)
                      
            f0_arr = gather_broadcast(f0_updated.vector().array())
            assign_to_vector(f0_new.vector(), f0_arr)
            
        
        else:
            f0_arr = gather_broadcast(f0.vector().array())
            assign_to_vector(f0_new.vector(), f0_arr)
        
    return f0_new
    
def vectorfield_to_components(u, S, dim):
    components = [df.Function(S) for i in range(dim)]
    assigners = [df.FunctionAssigner(S, u.function_space().sub(i)) for i in range(dim)]    
    for i, comp, assigner in zip(range(dim), components, assigners):
	assigner.assign(comp, u.sub(i))
    return components
    
def normalize_vector_field(u):
    dim = len(u)
    S = u.function_space().sub(0).collapse()
    
    components = vectorfield_to_components(u, S, dim)
    normarray = np.sqrt(sum(gather_broadcast(components[i].vector().array())**2 for i in range(dim)))
    
    for i in range(dim):    
	assign_to_vector(components[i].vector(), gather_broadcast(components[i].vector().array())/normarray)
	
    assigners = [df.FunctionAssigner(u.function_space().sub(i), S) for i in range(dim)]    
    for i, comp, assigner in zip(range(dim), components, assigners):
	assigner.assign(u.sub(i), comp)
    return u
    





def get_volume(geometry, u = None, chamber="lv"):

    if geometry.markers.has_key("ENDO"):
        lv_endo_marker = geometry.markers["ENDO"]
        rv_endo_marker = None
    else:
        lv_endo_marker = geometry.markers["ENDO_LV"]
        rv_endo_marker = geometry.markers["ENDO_RV"]

    marker = lv_endo_marker if chamber == "lv" \
             else rv_endo_marker

    if marker is None:
        return None
        

    if hasattr(marker, "__len__"):
        marker = marker[0]

    ds = df.Measure("exterior_facet",
                 subdomain_data = geometry.ffun,
                 domain = geometry.mesh)(marker)

    X = df.SpatialCoordinate(geometry.mesh)
    N = df.FacetNormal(geometry.mesh)

    if u is None:
        vol = df.assemble((-1.0/3.0)*df.dot(X,N)*ds)
    else:
        F = df.grad(u) + df.Identity(3)
        J = df.det(F)
        vol = df.assemble((-1.0/3.0)*df.dot(X + u, J*df.inv(F).T*N)*ds)
        

    return vol
def update_material_parameters(material_parameters, mesh):
    
    from .. import RegionalParameter
    
    new_matparams = {}
    for k,v in material_parameters.iteritems():
        if isinstance(v, RegionalParameter):
            meshfunction = df.MeshFunction("size_t", mesh, 3,
                                        mesh.domains())

            v_new = RegionalParameter(meshfunction)
            v_arr = gather_broadcast(v.vector().array())
            assign_to_vector(v_new.vector(), v_arr)
            new_matparams[k] = v_new

                           
        elif isinstance(v, df.Function):
            v_new = df.Function(df.FunctionSpace(mesh, v.function_space().ufl_element()))
            v_arr = gather_broadcast(v.vector().array())
            assign_to_vector(v_new.vector(), v_arr)
            new_matparams[k] = v_new

                   
        else:
            new_matparams[k] = v

    return new_matparams
def load_opt_target(h5name, h5group, key = "volume", data = "simulated"):
     
     
    with h5py.File(h5name) as f:
        vols = [a[:][0] for a in f[h5group]["passive_inflation"][key][data].values()]

    return vols
    
def save_unloaded_geometry(new_geometry, h5name, h5group, backward_displacement=None):



    fields = ['fiber', 'sheet', 'sheet_normal']
    local_basis = ['circumferential','radial', 'longitudinal']

    new_fields=[]
    for fattr in fields:
        if hasattr(new_geometry, fattr) and getattr(new_geometry, fattr) is not None:
            f = getattr(new_geometry, fattr).copy()
            f.rename(fattr, "microstructure")
            new_fields.append(f)
            
            new_local_basis=[]
    for fattr in local_basis:
        if hasattr(new_geometry, fattr) and getattr(new_geometry, fattr) is not None:
            f = getattr(new_geometry, fattr).copy()
            f.rename(fattr, "local_basis_function")
            new_local_basis.append(f)
            


    logger.debug("Save geometry to {}:{}".format(h5name,h5group))


    if backward_displacement:
        other_functions={"backward_displacement": backward_displacement}
    else:
        other_functions={}

    from mesh_generation.mesh_utils import save_geometry_to_h5
    save_geometry_to_h5(new_geometry.mesh, h5name, 
                        h5group, new_geometry.markers,
                        new_fields, new_local_basis,
                        other_functions=other_functions)


    

def continuation_step(params, it_, paramvec):

    # Use data from the two prevoious steps and continuation
    # to get a good next gues
    values = []
    vols = []

    v_target = load_opt_target(params["sim_file"], "0", "volume", "target")
    for it in range(it_):
        p_tmp = df.Function(paramvec.function_space())
        load_material_parameter(params["sim_file"], str(it), p_tmp)

        values.append(gather_broadcast(p_tmp.vector().array()))

        v = load_opt_target(params["sim_file"], str(it), "volume", "simulated")
        vols.append(v)

     
    ed_vols = np.array(vols).T[-1]
    # Make continuation approximation
    delta = (v_target[-1] - ed_vols[-2])/(ed_vols[-1] - ed_vols[-2])
    a_cont = (1-delta)*values[-2] + delta*values[-1]
    a_prev = values[-1]
    
        
    # Make sure next step is not to far away
    if hasattr(a_cont, "__len__"):
        
        a_next = np.array([min(max(a_cont[i], a_prev[i]/2), a_prev[i]*2) for i in range(len(a_cont))])


        # Just make sure that we are within the given bounds
        a = np.array([min(max(a_next[i], params["Optimization_parameters"]["matparams_min"]),
                          params["Optimization_parameters"]["matparams_max"]) for i in range(len(a_cont))])
        

    else:

        a_next = min(max(a_cont, a_prev/2), a_prev*2)
        
        # Just make sure that we are within the given bounds
        a = min(max(a_next, params["Optimization_parameters"]["matparams_min"]),
                params["Optimization_parameters"]["matparams_max"])
        
                
            

    print "#"*40
    print "delta = ", delta
    print "a_prev = ", a_prev
    print "a_next = ", a_next
    print "a_cont  = ", a_cont
    print "#"*40
    
    assign_to_vector(paramvec.vector(), a)
    
def load_material_parameter(h5name, h5group, paramvec):
    logger.info("Load {}:{}".format(h5name, h5group))
    group = "/".join([h5group, "passive_inflation", "optimal_control"])
    with df.HDF5File(df.mpi_comm_world(), h5name, "r") as h5file:
        h5file.read(paramvec, group)

