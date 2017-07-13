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
from dolfin import *
import numpy as np
import operator as op
import math

import matplotlib.pyplot as plt

from pulse_adjoint import LVTestPatient
from pulse_adjoint.setup_parameters import setup_application_parameters, setup_general_parameters
from pulse_adjoint.setup_optimization import make_solver_params
from pulse_adjoint.lvsolver import LVSolver3Field
from pulse_adjoint.iterate import iterate



####################
### Gernal setup ###
####################
setup_general_parameters()

PETScOptions.set('ksp_type', 'preonly')
PETScOptions.set('pc_factor_mat_solver_package', 'mumps')
PETScOptions.set("mat_mumps_icntl_7", 6)    


############
### MESH ###
############
# meshname = "benchmark"
meshname = "simple_ellipsoid"
patient = LVTestPatient(meshname)
mesh = patient.mesh
X = SpatialCoordinate(mesh)
N = FacetNormal(mesh)



# Cycle lenght
BCL = 200
t = 0.0
# Time increment
dt = 3.0

# End-Diastolic volume
ED_vol = 53.0

#####################################
# Parameters for Windkessel model ###
#####################################

# Aorta compliance (reduce)
Cao = 10.0/1000.0;
# Venous compliace
Cven = 400.0/1000.0;
# Dead volume
Vart0 = 510;
Vven0 = 2800;
# Aortic resistance
Rao = 10*1000.0 ;
Rven = 2.0*1000.0;
# Peripheral resistance (increase)
Rper = 10*1000.0;

V_ven = 3660 
V_art = 640

# scale geometry to match hemodynamics parameters
if meshname == "benchmark":
    mesh.coordinates()[:] /= 4.0
elif meshname == "simple_ellipsoid":
    mesh.coordinates()[:] *= 2.4


######################
### Material model ###
######################
material_model =  "guccione"
Ccoeff = 200. 
bf = 90.
bfs = 40.
bt = 10.

####################
### Active model ###
####################
active_model = "active_stress"

if active_model == "active_stress":
    T_ref = 60e3
else:
    T_ref = 0.3




##############
### OUTPUT ###
##############
dir_results = "results"
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

disp_file = XDMFFile(mpi_comm_world(), "{}/displacement.xdmf".format(dir_results))
pv_data  = {"pressure":[], "volume":[]}
output = "/".join([dir_results, "output_{}_ed{}.h5".format(meshname, ED_vol)])




########################
### Setup Parameters ###
########################

params = setup_application_parameters(material_model)

params["Material_parameters"]["C"] = Ccoeff
params["Material_parameters"]["bf"] = bf
params["Material_parameters"]["bfs"] = bfs
params["Material_parameters"]["bt"] = bt

params["base_bc"] = "fix_x"
params["base_spring_k"] = 1.0
# params["base_bc"] = "fixed"
params["active_model"] = active_model
params["T_ref"] = T_ref
params["gamma_space"] = "R_0"


######################
### Initialization ###
######################

# Solver paramters
solver_parameters, _, _ = make_solver_params(params, patient)

# Cavity volume
V0 = Expression("vol",vol = 0, name = "Vtarget", degree=1)
solver_parameters["volume"] = V0

# Solver
solver = LVSolver3Field(solver_parameters, use_snes=True)
set_log_active(True)
solver.parameters["solve"]["snes_solver"]["report"] =True
solver.parameters["solve"]["snes_solver"]['maximum_iterations'] = 50

# Surface measure
ds = Measure("exterior_facet", domain = solver.parameters["mesh"],
             subdomain_data = solver.parameters["facet_function"])
dsendo = ds(solver.parameters["markers"]["ENDO"][0])

# Set cavity volume
V0.vol = assemble(solver._V_u*dsendo)

# Initial solve
solver.solve()

# Save initial state
w = solver.get_state()
u, p, pinn = w.split(deepcopy=True)
U_save = Function(u.function_space(), name = "displacement")
U_save.assign(u)
disp_file.write(U_save)
pv_data["pressure"].append(float(pinn)/1000.0)
pv_data["volume"].append(V0.vol)



# Active contraction
from force import ca_transient
V_real = FunctionSpace(mesh, "R", 0)
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
    
    with HDF5File(mpi_comm_world(), output, "r") as h5file:
        if h5file.has_dataset("inflation"):
            h5file.read(solver.get_state(), "inflation")
            print("\nInflation phase fetched from output file.")
            inflate = False

if inflate:
    print("\nInflate geometry to volume : {}\n".format(ED_vol))
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


        print "V = ", control_values[i]
        print "P = {}".format(float(pinni)/1000.0)
        
        pv_data["pressure"].append(float(pinni)/1000.0)
        pv_data["volume"].append(control_values[i])

    
    with HDF5File(mpi_comm_world(), output, "w") as h5file:
        h5file.write(solver.get_state(), "inflation")
        

# Store ED solution
w = solver.get_state()
u, p, pinn = w.split(deepcopy=True)
U_save.assign(u)
disp_file.write(U_save)
pv_data["pressure"].append(float(pinn)/1000.0)
pv_data["volume"].append(ED_vol)
  
print("\nInflation succeded! Current pressure: {} kPa\n\n".format(float(pinn)/1000.0))


    
#########################
### Closed loop cycle ###
#########################

while (t < BCL):

    w = solver.get_state()
    u, p, pinn = w.split(deepcopy=True)
    
    p_cav = float(pinn)
    V_cav = assemble(solver._V_u*dsendo)
    if t + dt > BCL:
        dt = BCL - t
    t = t + dt

    target_gamma = ca_transient(t)


    # Update windkessel model
    Part = 1.0/Cao*(V_art - Vart0);
    Pven = 1.0/Cven*(V_ven - Vven0);
    PLV = float(p_cav);

    print "PLV = ", PLV
    print "Part = ", Part 
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
        _, states = iterate("gamma", solver, target_gamma, gamma, initial_number_of_steps = 1)
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
    print(msg)

          

fig = plt.figure()
ax = fig.gca()
ax.plot(pv_data["volume"], pv_data["pressure"])
ax.set_ylabel("Pressure (kPa)")
ax.set_xlabel("Volume (ml)")


fig.savefig("/".join([dir_results, "pv_loop.png"]))
plt.show()
