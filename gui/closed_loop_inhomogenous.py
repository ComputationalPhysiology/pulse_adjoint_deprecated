"""
This script implement a closed loop for a left ventricular geometry.
The parameters for the Windkessel model (the model for the ejection phase) 
has to be tuned corretly to get the right shape of the loop. 

You can change the geometry by changing the meshname, 
the end-diastolic volume by changing 'ED_vol' and the ejection fraction 
by chaning the Windkessel parmeters. 

You may also change the material model, the model for the active contraction as well
as the contractility in different segment according to AHA zones. 

"""


import sys, os
import dolfin as df
import numpy as np
import operator as op
import math

import matplotlib.pyplot as plt

from pulse_adjoint import LVTestPatient
from pulse_adjoint.setup_parameters import setup_application_parameters, setup_general_parameters
from pulse_adjoint.setup_optimization import make_solver_params, RegionalParameter, check_patient_attributes
from pulse_adjoint.lvsolver import LVSolver3Field
from pulse_adjoint.iterate import iterate

from pulse_adjoint.adjoint_contraction_args import logger



def closed_loop(parameters, advanced_parameters, CL_parameters):
    
    
    ####################
    ### Gernal setup ###
    ####################
    setup_general_parameters()
    
    df.PETScOptions.set('ksp_type', 'preonly')
    df.PETScOptions.set('pc_factor_mat_solver_package', 'mumps')
    df.PETScOptions.set("mat_mumps_icntl_7", 6)    
    
    
    ############
    ### MESH ###
    ############
    patient = parameters['patient']
    meshname = parameters['mesh_name']
    mesh = parameters['mesh']
    mesh = patient.mesh
    X = df.SpatialCoordinate(mesh)
    N = df.FacetNormal(mesh)
    
    
    
    # Cycle lenght
    BCL = CL_parameters['BCL']
    t = CL_parameters['t']
    # Time increment
    dt = CL_parameters['dt']
    
    # End-Diastolic volume
    ED_vol = CL_parameters['ED_vol']
    
    #####################################
    # Parameters for Windkessel model ###
    #####################################
    
    # Aorta compliance (reduce)
    Cao = CL_parameters['Cao']
    # Venous compliace
    Cven = CL_parameters['Cven']
    # Dead volume
    Vart0 = CL_parameters['Vart0']
    Vven0 = CL_parameters['Vven0']
    # Aortic resistance
    Rao = CL_parameters['Rao']
    Rven = CL_parameters['Rven']
    # Peripheral resistance (increase)
    Rper = CL_parameters['Rper']
    
    V_ven = CL_parameters['V_ven']
    V_art = CL_parameters['V_art']
    
    # scale geometry to match hemodynamics parameters
    mesh.coordinates()[:] *= CL_parameters['mesh_scaler']
    
    
    ######################
    ### Material model ###
    ######################
    material_model =  advanced_parameters['material_model']
    
    ####################
    ### Active model ###
    ####################
    active_model = advanced_parameters['active_model']
    T_ref = advanced_parameters['T_ref']
    
    # These can be used to adjust the contractility
    gamma_base = parameters['gamma']['gamma_base']
    gamma_mid =  parameters['gamma']['gamma_mid']
    gamma_apical =  parameters['gamma']['gamma_apical']
    gamma_apex = parameters['gamma']['gamma_apex']
    gamma_arr = np.array(gamma_base + gamma_mid + gamma_apical + gamma_apex)
    # gamma_arr = np.ones(17)
    
    
    ##############
    ### OUTPUT ###
    ##############
    dir_results = "results"
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    disp_file = df.XDMFFile(df.mpi_comm_world(), "{}/displacement.xdmf".format(dir_results))

    pv_data  = {"pressure":[], "volume":[]}
    output = "/".join([dir_results, "output_{}_ed{}.h5".format(meshname, ED_vol)])
    
    
    G = RegionalParameter(patient.sfun)
    G.vector()[:] = gamma_arr
    G_ = df.project(G.get_function(), G.get_ind_space())
    f_gamma = df.XDMFFile(df.mpi_comm_world(), "{}/activation.xdmf".format(dir_results))
    f_gamma.write(G_)
    
    
    
    ########################
    ### Setup Parameters ###
    ########################
    params = setup_application_parameters(material_model)
    params.remove("Material_parameters")
    matparams = df.Parameters("Material_parameters")
    for k, v in  advanced_parameters['mat_params'].iteritems():
        matparams.add(k,v)
    params.add(matparams)
   
   
   
    params["base_bc"] = parameters['BC_type']
    params["base_spring_k"] = advanced_parameters['spring_constant']
    # params["base_bc"] = "fixed"
    params["active_model"] = active_model
    params["T_ref"] = T_ref
    params["gamma_space"] = "regional"
    
    
    ######################
    ### Initialization ###
    ######################
    
    # Solver paramters
    check_patient_attributes(patient)
    solver_parameters, _, _ = make_solver_params(params, patient)
    
    # Cavity volume
    V0 = df.Expression("vol",vol = 0, name = "Vtarget", degree=1)
    solver_parameters["volume"] = V0
    
    # Solver
    solver = LVSolver3Field(solver_parameters, use_snes=True)
    df.set_log_active(True)
    solver.parameters["solve"]["snes_solver"]["report"] =True
    solver.parameters["solve"]["snes_solver"]['maximum_iterations'] = 50
    
    # Surface measure
    ds = df.Measure("exterior_facet", domain = solver.parameters["mesh"],
                 subdomain_data = solver.parameters["facet_function"])
    dsendo = ds(solver.parameters["markers"]["ENDO"][0])
    
    # Set cavity volume
    V0.vol = df.assemble(solver._V_u*dsendo)
    
    print V0.vol
    
    # Initial solve
    solver.solve()
    
    # Save initial state
    w = solver.get_state()
    u, p, pinn = w.split(deepcopy=True)
    U_save = df.Function(u.function_space(), name = "displacement")
    U_save.assign(u)
    disp_file.write(U_save)      
    
    file_format = "a" if os.path.isfile('pv_data_plot.txt') else "w"
    pv_data_plot = open('pv_data_plot.txt', file_format)
    pv_data_plot.write('{},'.format(float(pinn)/1000.0))
    pv_data_plot.write('{}\n'.format(V0.vol))

    pv_data["pressure"].append(float(pinn)/1000.0)
    pv_data["volume"].append(V0.vol)
    
    
    
    # Active contraction
    from force import ca_transient
    V_real = df.FunctionSpace(mesh, "R", 0)
    gamma = solver.parameters["material"].get_gamma()
    
    # times = np.linspace(0,200,200)
    # target_gamma = ca_transient(times)
    # plt.plot(target_gamma)
    # plt.show()
    # exit()g
    
    #######################
    ### Inflation Phase ###
    #######################
    inflate = True
    # Check if inflation allready exist
    if os.path.isfile(output):
        
        with df.HDF5File(df.mpi_comm_world(), output, "r") as h5file:
            if h5file.has_dataset("inflation"):
                h5file.read(solver.get_state(), "inflation")
                print ("\nInflation phase fetched from output file.")
                inflate = False
    
    if inflate:
        print ("\nInflate geometry to volume : {}\n".format(ED_vol))
        initial_step =  int((ED_vol - V0.vol) / 10.0) +1
        control_values, prev_states = iterate("expression", solver, V0, "vol",
                                              ED_vol, continuation=False,
                                              initial_number_of_steps=initial_step,
                                              log_level=10)
    
        # Store outout
        for i, wi in enumerate(prev_states):
            ui, pi, pinni = wi.split(deepcopy=True)
            U_save.assign(ui)
            disp_file.write(U_save)
    
            print ("V = {}".format(control_values[i]))
            print ("P = {}".format(float(pinni)/1000.0))
            
            pv_data_plot.write('{},'.format(float(pinni)/1000.0))
            pv_data_plot.write('{}\n'.format(control_values[i]))
            pv_data["pressure"].append(float(pinni)/1000.0)
            pv_data["volume"].append(control_values[i])
    
        
        with df.HDF5File(df.mpi_comm_world(), output, "w") as h5file:
            h5file.write(solver.get_state(), "inflation")
            
    
    # Store ED solution
    w = solver.get_state()
    u, p, pinn = w.split(deepcopy=True)
    U_save.assign(u)
    disp_file.write(U_save)
    
    pv_data_plot.write('{},'.format(float(pinn)/1000.0))
    pv_data_plot.write('{}\n'.format(ED_vol))
    pv_data["pressure"].append(float(pinn)/1000.0)
    pv_data["volume"].append(ED_vol)
      
    print ("\nInflation succeded! Current pressure: {} kPa\n\n".format(float(pinn)/1000.0))
    
    pv_data_plot.close()
        
    #########################
    ### Closed loop cycle ###
    #########################
    
    while (t < BCL):
    
        w = solver.get_state()
        u, p, pinn = w.split(deepcopy=True)
        
        p_cav = float(pinn)
        V_cav = df.assemble(solver._V_u*dsendo)
        if t + dt > BCL:
            dt = BCL - t
        t = t + dt
    
        target_gamma = ca_transient(t)
    
    
        # Update windkessel model
        Part = 1.0/Cao*(V_art - Vart0);
        Pven = 1.0/Cven*(V_ven - Vven0);
        PLV = float(p_cav);
    
        print ("PLV = {}".format(PLV))
        print ("Part = {}".format(Part))
        # Flux trough aortic valve
        if(PLV <= Part):
             Qao = 0.0;
        else:
             Qao = 1.0/Rao*(PLV - Part);
        
        # Flux trough mitral valve
        if(PLV >= Pven):
            Qmv = 0.0;
        else: 
            Qmv = 1.0/Rven*(Pven - PLV);
        
    
        Qper = 1.0/Rper*(Part - Pven);
    
        V_cav = V_cav + dt*(Qmv - Qao);
        V_art = V_art + dt*(Qao - Qper);
        V_ven = V_ven + dt*(Qper - Qmv);
    
        
    
        # Update cavity volume
        V0.vol = V_cav 
    
    
        # Iterate active contraction
        if t <= 150:
            target_gamma_ = target_gamma * gamma_arr
            _, states = iterate("gamma", solver, target_gamma_, gamma, initial_number_of_steps = 1)
        else:
            solver.solve()
        
    
        # Adapt time step
        if len(states) == 1:
            dt *= 1.7
        else:
            dt *= 0.5
              
        dt = min(dt, 10)
    
        # Store data 
        ui, pi, pinni = solver.get_state().split(deepcopy=True)
        U_save.assign(ui)
        disp_file.write(U_save)
    
        Pcav = float(pinni)/1000.0
        pv_data_plot = open('pv_data_plot.txt', 'a')
        pv_data_plot.write('{},'.format(Pcav))
        pv_data_plot.write('{}\n'.format(V_cav))
        pv_data_plot.close()
        pv_data["pressure"].append(Pcav)
        pv_data["volume"].append(V_cav)
        
        msg = ("\n\nTime:\t{}".format(t) + \
               "\ndt:\t{}".format(dt) +\
               "\ngamma:\t{}".format(target_gamma) +\
               "\nV_cav:\t{}".format(V_cav) + \
               "\nV_art:\t{}".format(V_art) + \
               "\nV_ven:\t{}".format(V_ven) + \
               "\nPart:\t{}".format(Part) + \
               "\nPven:\t{}".format(Pven) + \
               "\nPLV:\t{}".format(Pcav) + \
               "\nQper:\t{}".format(Qper) + \
               "\nQao:\t{}".format(Qao) + \
               "\nQmv:\t{}\n\n".format(Qmv))
        print ((msg))        
    
#==============================================================================
#     fig = plt.figure()
#     ax = fig.gca()
#     ax.plot(pv_data["volume"], pv_data["pressure"])
#     ax.set_ylabel("Pressure (kPa)")
#     ax.set_xlabel("Volume (ml)")
#     
#     
#     fig.savefig("/".join([dir_results, "pv_loop.png"]))
#     plt.show()
#==============================================================================
    return
#import threading
#thread1 = threading.Thread(target = closed_loop)
#thread1.start()

    