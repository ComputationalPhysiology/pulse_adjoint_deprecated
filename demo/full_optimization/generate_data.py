#!/usr/bin/env python
# Copyright (C) 2017 Henrik Finsberg
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
import os, yaml, shutil
import numpy as np
from dolfin import *


import pulse_adjoint.models.material as mat
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.utils import QuadratureSpace
from pulse_adjoint.setup_optimization import (setup_general_parameters,
                                              setup_adjoint_contraction_parameters,
                                              check_patient_attributes,
                                              make_solver_params)


from pulse_adjoint.numpy_mpi import*
from pulse_adjoint.optimization_targets import (OptimizationTarget, Regularization,
                                                RegionalStrainTarget, VolumeTarget)

from pulse_adjoint.adjoint_contraction_args import PHASES, logger

from mesh_generation import load_geometry_from_h5, save_geometry_to_h5
from mesh_generation.strain_regions import make_crl_basis

from pulse_adjoint import LVTestPatient

setup_general_parameters()

# This is temporary
foc = 1.54919333848


def setup(case,
          active_model,
          eps_vol,
          eps_strain,
          pressures,
          unload,
          base_spring_k,
          gamma_space, 
          isotropic, approx, restart):
    
    msg= ("\n\n"+" Test optimization on synthetic data  ".center(72, "#")+
          "\n\n\tcase = {}\n".format(case) + \
          "\tactive_model = {}\n".format(active_model) + \
          "\teps (volume) = {}\n".format(eps_vol) + \
          "\teps (strain) = {}\n".format(eps_strain) + \
          "\tunload = {}\n".format(unload) + \
          "\tbase_spring = {}\n".format(base_spring_k) + \
          "\tgamma space = {}\n".format(gamma_space) + \
          "\tisotrpoic = {}\n".format(isotropic) + \
          "\tapproximaion = {}\n\n".format(approx) + \
          "".center(72, "#") + "\n")
    print(msg)

    
    

    # This is the target matierial parameter
    passive_expr = Expression("5.93")
    matparams = get_matparams(isotropic)
    
    # This is the target contraction parameter
    active_expr =  Expression("0.1*(1+0.3*(x[2]+x[1]+x[0]))")

    # Application parameters
    ap_params = get_application_parameters(gamma_space,
                                           active_model,
                                           base_spring_k,
                                           matparams, restart)


    ap_params["volume_approx"] = approx
    
   
    geo, basis, f0 = create_geometry(case)
    
    # Make the solver parameters
    params, pressure_expr, control = make_solver_parameters(geo, ap_params)


    
    # params["passive_filling_duration"] = geo.passive_filling_duration
    params["basis"] = basis
    params["f0"] = f0
    

    params["active_model"] = active_model

    params["markers"] = geo.markers
    params["material_parameters"] = matparams

    parameters["adjoint"]["stop_annotating"] = True
    strains, vols,  ws, us, ps = generate_data(passive_expr, active_expr, params, ap_params,
                                               pressure_expr, eps_vol,
                                               eps_strain, pressures)


    if unload:
        params, ap_params, u_img, strains = create_unloaded_geometry(case, params, us, ap_params)
    else:
        h5name = "data/geometry_{}.h5".format(case)
        if os.path.isfile(h5name):
            os.remove(h5name )

        shutil.copy("data/geometry_{}_original.h5".format(case), h5name)
        u_img = None
    
    
    return (params, strains, vols, ap_params, passive_expr, active_expr, u_img)


def create_geometry(case):

    geo = LVTestPatient(case)
    mesh = geo.mesh

    e_circ = geo.circumferential
    e_long = geo.longitudinal
    e_rad = geo.radial

    basis = {"circumferential":e_circ, "longitudinal": e_long, "radial": e_rad}

    if not os.path.exists("data"):
        os.makedirs("data")

    shutil.copy(geo.paths["mesh_path"], "data/geometry_{}_original.h5".format(case))



    return geo, basis, geo.fiber

def create_unloaded_geometry(case, params, us, ap_params):

    # Create new loaded geometry
    V = VectorFunctionSpace(params["mesh"], "CG", 1)
        
    if ap_params["volume_approx"] == "project":
        u_img = project(us[1], V)
    else:
        #interpolate
        u_img = interpolate(us[1], V)


    # Recompute strains wrt to new reference
    F_ref = grad(u_img) + Identity(3)
    dX = Measure("dx",
                 subdomain_data = params["mesh_function"],
                 domain = params["mesh"])
    target_strain = RegionalStrainTarget(params["mesh"],
                                         params["basis"],
                                         dX,
                                         F_ref = F_ref,
                                         tensor = "gradu")
    new_strains = []
    target_strain.set_target_functions()
    for ui in us:
        target_strain.assign_simulated(ui)

        strain = [gather_broadcast(target_strain.simulated_fun[i].vector().array()) \
                  for i in range(target_strain.nregions)]
        new_strains.append(strain)

        
        
    mesh_img = Mesh(params["mesh"])
    ALE.move(mesh_img, u_img)
    
    # local basis
    c,r,l = make_crl_basis(mesh_img, foc)
    basis = {"circumferential":c, "longitudinal": l, "radial": r}
    
    
    # Fibers
    from unloading.utils import update_vector_field
    f0 = update_vector_field(params["f0"], mesh_img, u_img)

    
    ffun_img = MeshFunction("size_t", mesh_img, 2)
    ffun_img.array()[:] = params["facet_function"].array()

        
    geo_img = lambda : None
    geo_img.mesh = mesh_img
    geo_img.ffun = ffun_img
    geo_img.markers = params["markers"]
    geo_img.fiber = f0

    N = FacetNormal(geo_img.mesh)
    X = SpatialCoordinate(geo_img.mesh)
    ds = Measure("exterior_facet",
                 subdomain_data = geo_img.ffun,
                 domain = geo_img.mesh)(30)
    vol = assemble((-1.0/3.0)*dot(X,N)*ds)


    
    h5name = "data/geometry_{}.h5".format(case)
    if os.path.isfile(h5name):
        os.remove(h5name )
    
    save_geometry_to_h5(mesh_img, h5name, "", params["markers"],
                        [f0], [c,r,l], other_functions={"u_img":u_img})
    
    ap_params["Patient_parameters"]["mesh_group"] =""
    ap_params["Patient_parameters"]["mesh_path"] = h5name
    ap_params["passive_weights"] = "all"
    ap_params["unload"] = True
    ap_params["Unloading_parameters"]["tol"] = 1e-6
    ap_params["Unloading_parameters"]["maxiter"] = 40
    ap_params["Unloading_parameters"]["method"] = "fixed_point"
    
    ap_params["Unloading_parameters"]["unload_options"]["maxiter"] = 40
    ap_params["Unloading_parameters"]["unload_options"]["tol"] = 1e-10
    ap_params["Optimization_parameters"]["passive_maxiter"] = 40
  
    params, p_lv, control = make_solver_parameters(geo_img, ap_params)
    params["basis"] = basis
    params["f0"] = f0
    parameters["adjoint"]["stop_annotating"] = False

    return params, ap_params, u_img, new_strains


def get_matparams(isotropic = True):

    # Material coefficients
    matparams = {"a":2.28, "a_f":1.685, "b":9.726, "b_f":15.779}

    if isotropic:
        matparams["a_f"] = 0.0

 
    return matparams
        


def make_solver_parameters(geo, ap_params):

    
    check_patient_attributes(geo)
    solver_parameters, p_lv, control = make_solver_params(ap_params, geo)
    return solver_parameters, p_lv, control




def generate_data(passive_expr, active_expr,  params, ap_params, pressure_expr, 
                  eps_vol = 0.0, eps_strain = 0.0, pressures = [0.3]):


    logger.info("\n\nGenerate synthetic data....\n")

    
    # Create an object for each single material parameter
  
    family, degree = ap_params["matparams_space"].split("_")
    matparams_space = FunctionSpace(params["mesh"], family, int(degree))
      
    # The linear isotropic parameter
    a = interpolate(passive_expr, matparams_space)
    params["material_parameters"]["a"] = a

    gamma_family, gamma_degree = ap_params["gamma_space"].split("_")
    gamma_space = FunctionSpace(params["mesh"], gamma_family, int(gamma_degree))

    gamma = Function(gamma_space)
    act = interpolate(active_expr, gamma_space)
    

    
    material = mat.HolzapfelOgden(params["f0"], gamma,
                                  params["material_parameters"],
                                  **ap_params)
    params["material"] = material


    
    solver = LVSolver(params)
    solver.solve()

    
    us_vol = []
    us_strain = []
    ps = []
    ws = []

    from pulse_adjoint.iterate import iterate

    V_cg1 = VectorFunctionSpace(params["mesh"], "CG", 1)
    for it, pres in enumerate(pressures):

        
        iterate("pressure", solver, pres, pressure_expr)


        if it == len(pressures)-1:
            iterate("gamma", solver, act, gamma)
            
            if not ap_params["gamma_space"] == "R_0":
                ap_params["volume_approx"] = "project"

                
        w = solver.get_state()
        u,p = w.split(deepcopy=True)


        if ap_params["volume_approx"] == "project":
            us_vol.append(project(u.copy(True), V_cg1))
        elif ap_params["volume_approx"] == "interpolate":
            us_vol.append(interpolate(u.copy(True), V_cg1))
        else:
            us_vol.append(u.copy(True))

        if ap_params["strain_approx"] == "project":
            us_strain.append(project(u.copy(True), V_cg1))
        elif ap_params["strain_approx"] == "interpolate":
            us_strain.append(interpolate(u.copy(True), V_cg1))
        else:
            us_strain.append(u.copy(True))
            
        ps.append(p.copy(True))
        ws.append(w.copy(True))


        
    
    dX = Measure("dx",
                 subdomain_data = params["mesh_function"],
                 domain = params["mesh"])
    target_strain = RegionalStrainTarget(params["mesh"],
                                         params["basis"],
                                         dX,
                                         tensor = "gradu")

    target_strain.set_target_functions()

    strains_arr = []
    vols = []
    logger.info("\n\t{:>10}\t{:>10}\t{:>10}".format("Pressure", "Volume", "Vol+noise"))
    for k, (u_vol, u_strain) in enumerate(zip(us_vol, us_strain)):

        if params["basis"]:
            # strains_zeros = []
            strains_arr_it = []
            target_strain.assign_simulated(u_strain)
            for i in range(target_strain.nregions):
                s = gather_broadcast(target_strain.simulated_fun[i].vector().array())
                
                noise = [0.0]*len(target_strain.crl_basis) if eps_strain <= 0 \
                        else np.random.normal(0,eps_strain, len(target_strain.crl_basis))
                s_tot = np.add(s, noise)

                strains_arr_it.append(s_tot)
            
            strains_arr.append(strains_arr_it)
        
        
        if params.has_key("markers"):
            dS = Measure("exterior_facet",
                         subdomain_data = params["facet_function"],
                         domain = params["mesh"])(params["markers"]["ENDO"][0])
        
            X = SpatialCoordinate(params["mesh"])
            N = FacetNormal(params["mesh"])
            vol_ref = assemble((-1.0/3.0)*dot(X, N)*dS)
            
            target_vol = VolumeTarget(params["mesh"], dS, "LV", "project")
            target_vol.set_target_functions()
            target_vol.assign_simulated(u_vol)
            vol = gather_broadcast(target_vol.simulated_fun.vector().array())[0]
            noise = 0.0 if eps_vol <= 0 else np.random.normal(0,eps_vol)
            vol_tot = vol+noise

            logger.info("\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}".format(pressures[k], vol, vol_tot))
            
            if len(vols) == 0:
                vol_prev = vol_ref
            else:
                vol_prev = vols[-1][-1]

            vols.append([vol_prev, vol_tot])

        else:
            vols = None

    logger.info("Done generating synthetic data.\n")
            
    return strains_arr, vols, ws, us_vol, ps
  

    



def get_application_parameters(gamma_space = "CG_1", 
                               active_model = "active_strain",
                               base_spring_k = 10.0,
                               matparams = None, restart = True):
    ap_params = setup_adjoint_contraction_parameters()

    
    if matparams:
        for k, v in matparams.iteritems():
            ap_params["Material_parameters"][k] = v

    ap_params["Optimization_parameters"]["passive_maxiter"] = 100
    ap_params["Optimization_parameters"]["passive_opt_tol"] = 1e-10
    ap_params["Optimization_parameters"]["active_maxiter"] = 100
    ap_params["Optimization_parameters"]["active_opt_tol"] = 1e-16

    ap_params["base_spring_k"] = base_spring_k
    ap_params["gamma_space"] = gamma_space
    ap_params["matparams_space"] = "R_0"

    ap_params["active_model"] = active_model

    if active_model == "active_stress":
        ap_params["T_ref"] = 500.0
    else:
        ap_params["T_ref"] = 0.5
        
    ap_params["log_level"] = 20
    
    return ap_params
    




def generate_synthetic_data(case, unload=True, active_model = "active_strain",
                            gamma_space = "CG_1", base_spring_k = 10.0, **kwargs):


    
    # Change these if you want to add some noise
    eps_strain = 0.0
    eps_vol = 0.0

    # Pressure (unloaded, image-based, end-diastole, active point)
    pressures = [0, 0.6, 0.8, 1.0]

    # Project or iterpolate the displacement before computing the volume.
    # When mesh is moved, this correspond to interpolation which is why we must
    # use interpolation when doing uloading. Note that dolfin-adjoint can only
    # registrer projection so if you need the gradient you better use project. 
    approx = "interpolate" if unload else "project"

    
   
    
    (params, strains, vols, ap_params,
     passive_expr, active_expr, u_img) \
        =setup(case = case,
            active_model = active_model,
            eps_vol = eps_vol,
            eps_strain = eps_strain,
            pressures = pressures,
            unload = unload,
            base_spring_k = base_spring_k,
            gamma_space = gamma_space, approx = approx,
            isotropic=False, restart = True)
    

    start_idx = 1 if unload else 0
    strain = {i+1:[] for i in range(len(strains[0]))}
    for s in strains[start_idx:]:
        for i,si in enumerate(s, start=1):
            strain[i].append(si.tolist())

    volume = np.array(vols).T[1][start_idx:]
    pressure = pressures[start_idx:]

    pdf = 2 if unload else 3
    data = {"passive_filling_duration": pdf,
            "pressure": pressure,
            "volume": volume.tolist(),
            "strain": strain}

    datafile = "data/measurements_{}.yml".format(case)
    if os.path.isfile(datafile):
        os.remove(datafile)
    
    with open(datafile, "wb") as f:
        yaml.dump(data, f, default_flow_style=False)


    

   

if __name__ == "__main__":
    
    

    case = "simple_ellipsoid"
    generate_synthetic_data(case)
