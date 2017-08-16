
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:37:33 2017

@author: sigurdll
"""

#General Solver
import dolfin as df
import numpy as np

from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.setup_parameters import setup_general_parameters, setup_material_parameters
from pulse_adjoint.setup_optimization import RegionalParameter
from pulse_adjoint import LVTestPatient
from pulse_adjoint.models import material as mat
from pulse_adjoint.iterate import iterate
from pulse_adjoint.utils import QuadratureSpace
from pulse_adjoint.adjoint_contraction_args import logger
from mesh_generation.mesh_utils import load_geometry_from_h5

import logging, sys, os

#==============================================================================
# patient = load_geometry_from_h5('simple_ellipsoid.h5')
# current_mesh = patient.mesh
# folder_path = '/home/sigurdll/Desktop/S17/finsberg-pulse_adjoint-cb5be41ee609/demo'
# mesh_name = 'simple_ellipsoid'
# lv_pressure = 5.0
# rv_pressure = 3.0
# cont_mult = 2.5
# 
# active_model = 'active_strain'
# material_model = 'holzapfel_ogden'
# spring_area = 'BASE'
# BC_type = 'Fixed in one direction'
# spring_constant = 1.0
# T_ref = 0.1
# 
# 
# parameters={'patient' : patient,
#             'mesh' : current_mesh,
#             'folder_path' : folder_path,
#             'mesh_name' : mesh_name,
#             'lv_pressure' : lv_pressure,
#             'rv_pressure' : rv_pressure,
#             'contraction_multiplier' : cont_mult}
#             
# advanced_parameters={'active_model' : active_model,
#                      'material_model' : material_model,
#                      'BC_type' : BC_type,
#                      'spring_area' : spring_area,
#                      'spring_constant' : spring_constant,
#                      'T_ref' : T_ref}
#==============================================================================

def general_solver(parameters, advanced_parameters):
    print 'TESTING THREAD1'
    setup_general_parameters()
    patient = parameters['patient']
    mesh = patient.mesh
    mesh_name = parameters['mesh_name']
    marked_mesh = patient.ffun
    N = df.FacetNormal(mesh)
    fibers = patient.fiber
    
    # Getting BC
    BC_type = parameters['BC_type']
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        if BC_type == 'fix_x':
            no_base_x_tran_bc = df.DirichletBC(V.sub(0), df.Constant(0.0), marked_mesh, patient.markers["BASE"][0])
        elif BC_type == 'fixed':
            no_base_x_tran_bc = df.DirichletBC(V, df.Constant((0.0, 0.0, 0.0)), marked_mesh, patient.markers["BASE"][0])
        return no_base_x_tran_bc
        
    
    # Contraction parameter
    # gamma = Constant(0.0)
    #gamma = df.Function(df.FunctionSpace(mesh, "R", 0))
    gamma = RegionalParameter(patient.sfun)
    
    gamma_values = parameters['gamma']
    gamma_base = gamma_values['gamma_base']
    gamma_mid =  gamma_values['gamma_mid']
    gamma_apical =  gamma_values['gamma_apical']
    gamma_apex = gamma_values['gamma_apex']
    

    gamma_arr = np.array(gamma_base + gamma_mid + gamma_apical + gamma_apex)

    G = RegionalParameter(patient.sfun)
    G.vector()[:] = gamma_arr
    G_ = df.project(G.get_function(), G.get_ind_space())
    
    f_gamma = df.XDMFFile(df.mpi_comm_world(), "activation.xdmf")
    f_gamma.write(G_)
    
    # Pressure
    pressure = df.Constant(0.0)
    lv_pressure = df.Constant(0.0)
    rv_pressure = df.Constant(0.0)
    #lv_pressure = df.Constant(0.0)
    #rv_pressure = df.Constant(0.0)

    # Spring
    spring_constant = advanced_parameters['spring_constant']
    spring = df.Constant(spring_constant)
    spring_area = advanced_parameters['spring_area']

    # Set up material model
    material_model = advanced_parameters['material_model']
    active_model = advanced_parameters['active_model']
    T_ref = advanced_parameters['T_ref']
    matparams=advanced_parameters['mat_params']
    #matparams=setup_material_parameters(material_model)
    if material_model == 'guccione':
        try:
            args = (fibers,
                    gamma,
                    matparams,
                    active_model,
                    patient.sheet,
                    patient.sheet_normal,
                    T_ref)
                    
        except AttributeError:
            print 'Mesh does not have "sheet" attribute, choose another material model.'
            return
            
    else:
        args = (fibers,
            gamma,
            matparams,
            active_model,
            #patient.sheet,
            #patient.sheet_normal,
            T_ref)        
    
    if material_model == 'holzapfel_ogden':
        material = mat.HolzapfelOgden(*args)

    elif material_model == 'guccione':
        material = mat.Guccione(*args)
        
    elif material_model == 'neo_hookean':
        material = mat.NeoHookean(*args)

    # Create parameters for the solver
    if mesh_name == 'biv_ellipsoid.h5':
        params= {"mesh": mesh,
                 "facet_function": marked_mesh,
                 "facet_normal": N,
                 "state_space": "P_2:P_1",
                 "compressibility":{"type": "incompressible",
                                    "lambda":0.0},
                 "material": material,
                 "bc":{"dirichlet": make_dirichlet_bcs,
                       "neumann":[[lv_pressure, patient.markers["ENDO_LV"][0]],
                                  [rv_pressure, patient.markers["ENDO_RV"][0]]],
                       "robin":[[spring, patient.markers[spring_area][0]]]}}
    else:
        params= {"mesh": mesh,
                 "facet_function": marked_mesh,
                 "facet_normal": N,
                 "state_space": "P_2:P_1",
                 "compressibility":{"type": "incompressible",
                                    "lambda":0.0},
                 "material": material,
                 "bc":{"dirichlet": make_dirichlet_bcs,
                       "neumann":[[pressure, patient.markers["ENDO"][0]]],
                       "robin":[[spring, patient.markers[spring_area][0]]]}}
 


    df.parameters["adjoint"]["stop_annotating"] = True

    # Initialize solver
    solver = LVSolver(params)
    print 'TESTING THREAD2'
    # Solve for the initial state
    folder_path = parameters['folder_path']
    u,p = solver.get_state().split(deepcopy=True)
    U = df.Function(u.function_space(),
                name ="displacement")
    f = df.XDMFFile(df.mpi_comm_world(), folder_path + "/displacement.xdmf")
    
    sigma_f = solver.postprocess().cauchy_stress_component(fibers)
    V = df.FunctionSpace(mesh, 'DG', 1)
    sf = df.project(sigma_f, V)
    SF = df.Function(sf.function_space(), name = 'stress')
    g = df.XDMFFile(df.mpi_comm_world(), folder_path + "/stress.xdmf")

    solver.solve()
    
    u1,p1 = solver.get_state().split(deepcopy=True)
    U.assign(u1)
    f.write(U)

    sigma_f1 = solver.postprocess().cauchy_stress_component(fibers)
    sf1 = df.project(sigma_f1, V)
    SF.assign(sf1)
    g.write(SF)    
    
    # Put on some pressure and solve
    plv = parameters['lv_pressure']
    prv = parameters['rv_pressure']
    
    if mesh_name == 'biv_ellipsoid.h5':
        iterate("pressure", solver, (plv, prv), {"p_lv":lv_pressure, "p_rv":rv_pressure})
    else:
        iterate("pressure", solver, plv, {"p_lv":pressure})
    
    u2,p2 = solver.get_state().split(deepcopy=True)
    U.assign(u2)
    f.write(U)

    sigma_f2 = solver.postprocess().cauchy_stress_component(fibers)
    sf2 = df.project(sigma_f2, V)
    SF.assign(sf2)
    g.write(SF)
    
    # Put on some active contraction and solve
    cont_mult = parameters['contraction_multiplier']
    g_ = cont_mult * gamma_arr
    
    iterate("gamma", solver, g_, gamma, max_nr_crash=100, max_iters=100)    
    u3,p3 = solver.get_state().split(deepcopy=True)
    U.assign(u3)
    f.write(U)

    sigma_f3 = solver.postprocess().cauchy_stress_component(fibers)
    sf3 = df.project(sigma_f3, V)
    SF.assign(sf3)
    g.write(SF)
    
    fname_u = "output_u.h5"
    if os.path.isfile(fname_u): os.remove(fname_u)
    u1.rename("u1", "displacement")
    u2.rename("u2", "displacement")
    u3.rename("u3", "displacement")

    fname_sf = "output_sf.h5"    
    if os.path.isfile(fname_sf): os.remove(fname_sf)
    sf1.rename("sf1", "stress")
    sf2.rename("sf2", "stress")
    sf3.rename("sf3", "stress")
    
    
    save_to_h5(fname_u, mesh, u1, u2, u3)
    save_to_h5(fname_sf, mesh, sf1, sf2, sf3)
   
    return u1, u2, u3, sf1, sf2, sf3
    
def save_to_h5(fname, mesh, *args, **kwargs):
    
     with df.HDF5File(df.mpi_comm_world(), fname, "w") as h5file:

        h5file.write(mesh, "mesh")       
        for i in args:
            h5file.write(i, "/".join([i.label(),i.name()]))


def load_from_h5(fname):
    
    with df.HDF5File(df.mpi_comm_world(), fname, "r") as h5file:
            
        mesh = df.Mesh()
        h5file.read(mesh, "mesh", True)
        
        out_list = []
        if fname == 'output_u.h5':
            ugroup = "displacement/u{}"
            i = 1
            while(h5file.has_dataset(ugroup.format(i))):
                
                el = df.VectorElement(df.FiniteElement('Lagrange', df.tetrahedron, 2), dim=3)                  
                V = df.FunctionSpace(mesh, el)
                u = df.Function(V)
                h5file.read(u, ugroup.format(i))
                u.rename("u{}".format(i), "displacement")
                out_list.append(u)
                
                i += 1
        elif fname == 'output_sf.h5':
            sfgroup = "stress/sf{}"
            i = 1
            while(h5file.has_dataset(sfgroup.format(i))):
                
                el = df.FiniteElement('Lagrange', df.tetrahedron, 2)                  
                V = df.FunctionSpace(mesh, el)
                sf = df.Function(V)
                h5file.read(sf, sfgroup.format(i))
                sf.rename("sf{}".format(i), "stress")
                out_list.append(sf)
                
                i += 1
    return out_list

    
#general_solver(parameters, advanced_parameters)