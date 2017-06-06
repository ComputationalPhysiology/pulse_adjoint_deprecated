"""
This script implements problem 2 and 3 from the 
cardiac mechanics benchmark [1].

[1] Land, Sander, et al. "Verification of cardiac 
mechanics software: benchmark problems and solutions 
for testing active and passive material behaviour." 
Proc. R. Soc. A. Vol. 471. No. 2184. The Royal Society, 2015.

"""

import dolfin as df
import numpy as np

from pulse_adjoint.setup_parameters import (setup_adjoint_contraction_parameters,
                                            setup_material_parameters, setup_general_parameters)
from pulse_adjoint.lvsolver import LVSolver
from patient_data import LVTestPatient
from pulse_adjoint.iterate import iterate_pressure, iterate_gamma
from pulse_adjoint.setup_optimization import make_solver_params
from pulse_adjoint.models.material import Guccione

setup_general_parameters()


def problem2():


    
    patient = LVTestPatient("benchmark")
    
    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    params["base_bc"] = "fixed"
    material_model = "guccione"


    
    solver_parameters, pressure, paramvec= make_solver_params(params, patient)
    V_real = df.FunctionSpace(solver_parameters["mesh"],  "R", 0)
    gamma = df.Function(V_real, name = "gamma")

    matparams = setup_material_parameters(material_model)

    matparams["C"] = 10.0
    matparams["bf"] = 1.0
    matparams["bt"] = 1.0
    matparams["bfs"] = 1.0
    
    args = (patient.fiber,
            gamma,
            matparams,
            "active_stress",
            patient.sheet,
            patient.sheet_normal,
            params["T_ref"])

  
    material = Guccione(*args)        
    solver_parameters["material"] = material


    
    solver = LVSolver(solver_parameters)
    solver.parameters["solve"]["snes_solver"]["report"] = True

    
    solver.solve()

    iterate_pressure(solver, 10.0, pressure)

    u,p = solver.get_state().split(deepcopy=True)

    f = df.XDMFFile(df.mpi_comm_world(), "benchmark_2.xdmf")
    f.write(u)


def problem3():


    
    patient = LVTestPatient("benchmark")
    
    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()
    params["phase"] == "all"
    active_model = "active_stress"
    params["active_model"] = active_model
    params["T_ref"] = 60.0
    params["base_bc"] = "fixed"

    # material_model = "guccione"
    material_model = "holzapfel_ogden"
    # material_model = "neo_hookean"

    
    solver_parameters, pressure, paramvec= make_solver_params(params, patient)
    V_real = df.FunctionSpace(solver_parameters["mesh"],  "R", 0)
    gamma = df.Function(V_real, name = "gamma")
    target_gamma = df.Function(V_real, name = "target gamma")
    
    
    matparams = setup_material_parameters(material_model)

    if material_model == "guccione":
        matparams["C"] = 2.0
        matparams["bf"] = 8.0
        matparams["bt"] = 2.0
        matparams["bfs"] = 4.0
       
    args = (patient.fiber,
            gamma,
            matparams,
            active_model,
            patient.sheet,
            patient.sheet_normal,
            params["T_ref"])


    if material_model == "guccione":
        material = Guccione(*args)
    else:
        material = HolzapfelOgden(*args)
        
    
    solver_parameters["material"] = material


    
    solver = LVSolver(solver_parameters)
    solver.parameters["solve"]["snes_solver"]["report"] = True

    
    solver.solve()

    p_end = 15.0
    g_end = 1.0

    N = 5
    df.set_log_active(True)
    df.set_log_level(df.INFO)
    f = df.XDMFFile(df.mpi_comm_world(), "benchmark_3.xdmf")
    u,p = solver.get_state().split(deepcopy=True)
    U = df.Function(u.function_space(), name ="displacement")
    
    for (plv, g) in zip(np.linspace(0,p_end, N),
                        np.linspace(0,g_end, N)):
    

        t = df.Timer("Test Material Model")
        t.start()
        iterate_pressure(solver, plv, pressure)
        

        u,p = solver.get_state().split(deepcopy=True)
        U.assign(u)
        f.write(U)

        
        target_gamma.assign(df.Constant(g))

        iterate_gamma(solver, target_gamma, gamma)
        
        
        u,p = solver.get_state().split(deepcopy=True)
        U.assign(u)
        f.write(U)

        


    
    

if __name__ == "__main__":
    problem2()
    # problem3()














