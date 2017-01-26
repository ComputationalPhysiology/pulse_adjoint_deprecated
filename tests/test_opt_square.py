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

from utils import *


def make_solver_params(mesh):

    DIR_BOUND = 1
    NEU_BOUND = 2

    dim = mesh.geometry().dim()
    
    N = FacetNormal(mesh)
    dir_sub = CompiledSubDomain("near(x[1], 0)")
    neu_sub = CompiledSubDomain("near(x[1], 1)")
    # Mark boundaries

    if dim == 2:
        ffun = MeshFunction("size_t", mesh, 1)
       
    
        def make_dirichlet_bcs(W):
            bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), ffun, DIR_BOUND)]
            return bcs
    else:
        ffun = MeshFunction("size_t", mesh, 2)
       
    
        def make_dirichlet_bcs(W):
            bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), ffun, DIR_BOUND)]
            return bcs
        
    ffun.set_all(0)
    dir_sub.mark(ffun, DIR_BOUND)
    neu_sub.mark(ffun, NEU_BOUND)


    # Dummy
    p_lv = Expression("t", t = 0.0, element = FiniteElement("R", mesh.ufl_cell(), 0))
    T = Constant(0.1)
    
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
             "passive_filling_duration": 1,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "base_bc_y":None,
             "base_bc_z":None,
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "solve":prm, 
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[T, NEU_BOUND]]}}

    return params, p_lv


def get_strain_markers_3d(mesh, nregions):
    strain_markers = MeshFunction("size_t", mesh, 3)
    strain_markers.set_all(0)
    
    xs = np.linspace(0,1,nregions+1)

    region = 0
    for it_x in range(nregions):
        for it_y in range(nregions):
            for it_z in range(nregions):
            
                region += 1
                
                domain_str = ""
            
                domain_str += "x[0] >= {}".format(xs[it_x])
                domain_str += " && x[1] >= {}".format(xs[it_y])
                domain_str += " && x[2] >= {}".format(xs[it_z])
                domain_str += " && x[0] <= {}".format(xs[it_x+1])
                domain_str += " && x[1] <= {}".format(xs[it_y+1])
                domain_str += " && x[2] <= {}".format(xs[it_z+1])
                print domain_str
            
            
                len_sub = CompiledSubDomain(domain_str)
                len_sub.mark(strain_markers, region)
            
    return strain_markers

def get_strain_markers_2d(mesh, nregions):
    strain_markers = MeshFunction("size_t", mesh, 2)
    strain_markers.set_all(0)
    
    xs = np.linspace(0,1,nregions+1)

    region = 0
    for it_x in range(nregions):
        for it_y in range(nregions):
            
            region += 1
            
            domain_str = ""
            
            domain_str += "x[0] >= {}".format(xs[it_x])
            domain_str += " && x[1] >= {}".format(xs[it_y])
            domain_str += " && x[0] <= {}".format(xs[it_x+1])
            domain_str += " && x[1] <= {}".format(xs[it_y+1])
            # print domain_str
            
            len_sub = CompiledSubDomain(domain_str)
            len_sub.mark(strain_markers, region)
    return strain_markers


def setup(phase, material_model, active_model,
          fiber_angle, ndiv,
          control_regions, strain_regions, dim):

    msg = ("phase = {}\n".format(phase) + \
           "material_model = {}\n".format(material_model) + \
           "active_model = {}\n".format(active_model) + \
           "fiber_angle = {}\n".format(fiber_angle) + \
           "dim = {}\n".format(dim) + \
           "ndiv= {}\n".format(ndiv) + \
           "control_regions = {}\n".format(control_regions) + \
           "strain_regions = {}\n".format(strain_regions))
    print(msg)

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


    # expr = Expression("sin(x[0])+2.0")
    if phase == "passive":
        expr = Expression("x[2]-x[1]+x[0]+2.0")
        initial_guess = 1.0
    else:
        expr =  Expression("0.1*(1.0+0.3*(x[2]+x[1]+x[0]))")
        initial_guess = 0.0
    



    mesh = UnitSquareMesh(ndiv, ndiv) if dim == 2 \
           else UnitCubeMesh(ndiv, ndiv, ndiv)
        
    V_cg1 = VectorFunctionSpace(mesh, "CG", 1)
    V_f = QuadratureSpace(mesh, 4)
    if dim == 2:
        
        strain_markers = get_strain_markers_2d(mesh, strain_regions)
        control_markers = get_strain_markers_2d(mesh, control_regions)
        regions = strain_regions**2

        e_circ = interpolate(Expression(("0.0", "1.0")), V_cg1)
        e_long = interpolate(Expression(("1.0", "0.0")), V_cg1)
        basis = {"e_circ":e_circ, "e_long": e_long}

        
        f0 = interpolate(Expression(fiber_angle), V_f)
        
    else:
       
        strain_markers = get_strain_markers_3d(mesh, strain_regions)
        control_markers = get_strain_markers_3d(mesh, control_regions)
        regions = strain_regions**3

        e_circ = interpolate(Expression(("0.0", "1.0", "0.0")), V_cg1)
        e_long = interpolate(Expression(("1.0", "0.0", "0.0")), V_cg1)
        e_rad = interpolate(Expression(("0.0", "0.0", "1.0")), V_cg1)
        basis = {"e_circ":e_circ, "e_long": e_long, "e_rad": e_long}

        f0 = interpolate(Expression(("1/sqrt(3)",  "1/sqrt(3)",  "1/sqrt(3)")), V_f)
        

 
    # Make the solver parameters
    params, p_lv = make_solver_params(mesh)



    params["basis"] = basis
    params["f0"] = f0
    params["strain_markers"] = strain_markers
    params["control_markers"] = control_markers
    params["nregions"] = regions
    params["active_model"] = active_model
    params["material_model"] = material_model
    # params["markers"] = geo.markers
    params["material_parameters"] = matparams
    params["dmu"] = Measure("dx", subdomain_data = strain_markers, domain = mesh)
    params["dmu_control"] = Measure("dx", subdomain_data = control_markers, domain = mesh)
    
    space = "regional"
    ap_params = get_application_parameters(space, phase, active_model)
    ap_params["Passive_optimization_weigths"]["regularization"] = 0.0
    ap_params["Passive_optimization_weigths"]["regional_strain"] = 1.0
    ap_params["Passive_optimization_weigths"]["volume"] = 0.0
    ap_params['Optimization_targets']['volume'] = False

    ap_params["Active_optimization_weigths"]["regularization"] = 0.0
    ap_params["Active_optimization_weigths"]["regional_strain"] = 1.0
    ap_params["Active_optimization_weigths"]["volume"] = 0.0
    
    ap_params["sim_file"] = "test_opt.h5"
   
   

    parameters["adjoint"]["stop_annotating"] = True
    strains, vols,  w, u, p, f_ex = generate_data(expr, params, ap_params, p_lv, phase)
    params, p_lv = make_solver_params(mesh)
    params["basis"] = basis
    params["f0"] = f0
    params["strain_markers"] = strain_markers
    params["control_markers"] = control_markers
    params["nregions"] = regions
    # params["markers"] = geo.markers
    params["active_model"] = active_model
    params["material_model"] = material_model
    params["material_parameters"] = matparams
    params["dmu"] = Measure("dx", subdomain_data = strain_markers, domain = mesh)
    params["dmu_control"] = Measure("dx", subdomain_data = control_markers, domain = mesh)
    parameters["adjoint"]["stop_annotating"] = False

    return params, strains, vols, ap_params, p_lv, w, u, p, f_ex, basis



def optimize():

    plot_sol = True

    phase = "active"
    material_model = "holzapfel_ogden"
    active_model = "active_strain"
    fiber_angle = ("0.0", "1.0")
    weight = (0.5,0.5)
    
    control_regions = 4
    strain_regions = 4
    ndiv = 4
    dim = 2


    params, strains, vols, ap_params, p_lv, w, u, p, f_ex, basis \
        =setup(phase = phase,
               material_model = material_model,
               active_model = active_model,
               fiber_angle = fiber_angle,
               dim = dim,
               control_regions = control_regions,
               strain_regions = strain_regions,
               ndiv = ndiv)
    
    print("Exact control = {}\n".format(f_ex.vector().array()))
    ap_params["passive_relax"] = 1.0
    ap_params["active_relax"] = 1.0
    rd,  forward_result, targets = run_optimization(params, strains, vols, ap_params, p_lv, 0.0, return_rd = False)
    
    
    sim_strains = np.array([gather_broadcast(v.array()) for v in targets["regional_strain"].results["simulated"][-1]])
    
    strains = np.array(strains)
                                         

    if plot_sol:
        
        f, ax = plt.subplots(1, len(basis.keys()))
        
        for i, k in enumerate(basis.keys()):
            ks = k.split("_")[-1]
            ax[i].set_title(ks)
   
            x = range(len(strains.T[i]))
            ax[i].plot(x, strains.T[i], "bo", label = "synthetic")
            ax[i].plot(x, sim_strains.T[i],  "ro", label = "simulated")
            ax[i].legend()
        plt.show()

    if phase == "active":

        paramvec = get_optimal_gamma(params, ap_params)
        space_str = "gamma_space"
    else:
        
        paramvec = get_optimal_matparam(params, ap_params)
        space_str = "matparams_space"
    
    if ap_params[space_str] == "regional":
        V = FunctionSpace(params["mesh"], "DG", 0)
        f_DG = project(f_ex.get_function(), V)
        
        paramvec_DG = project(paramvec.get_function(), V)

        if plot_sol:
            
            plot(paramvec_DG, mode = "color")
            plot(f_DG, mode="color")
            plot(paramvec_DG-f_DG, mode = "color")
            interactive()
        
    else:
        
        if plot_sol:
        
            plot(paramvec, mode = "color")
            plot(f_ex, mode="color")
            interactive()
            plot(paramvec-f_ex, mode = "color")
            interactive()

    


if __name__ == "__main__":
    setup_general_parameters()
    optimize()
