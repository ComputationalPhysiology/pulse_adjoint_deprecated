import dolfin as df
import numpy as np

from pulse_adjoint import LVTestPatient
from pulse_adjoint.setup_parameters import (setup_adjoint_contraction_parameters,
                                            setup_general_parameters)

from pulse_adjoint.setup_optimization import make_solver_params, get_material_model, get_volume

from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.models.material import HolzapfelOgden, Guccione, NeoHookean



setup_general_parameters()
patient = LVTestPatient()
patient.mesh.coordinates()[:] *= 3.15


def setup_material_parameters(material_model):
    """
    Choose parameters based on 
    
    Hadjicharalambous, Myrianthi, et al. "Analysis of passive 
    cardiac constitutive laws for parameter estimation using 3D 
    tagged MRI." Biomechanics and modeling in mechanobiology 14.4 
    (2015): 807-828.
    
    """
    material_parameters = df.Parameters("Material_parameters")
    
    if material_model == "guccione":
        material_parameters.add("C", 0.3)
        # material_parameters.add("C", 0.18)
        material_parameters.add("bf", 27.75)
        material_parameters.add("bt", 5.37)
        material_parameters.add("bfs", 2.445)

        

    elif material_model == "neo_hookean":
        
        material_parameters.add("mu", 5.6)
        # material_parameters.add("mu", 10.0)
        
    else:
        # material_model == "holzapfel_ogden":
        
        material_parameters.add("a", 0.80)
        # material_parameters.add("a", 4.0)
        material_parameters.add("a_f", 1.4)
        # material_parameters.add("a_f", 10.0)
        material_parameters.add("b", 5.0)
        material_parameters.add("b_f", 5.0)

    return material_parameters
    
def run(active_model, material_model, matparams_space):

    params = setup_adjoint_contraction_parameters(material_model)
    if active_model == "active_strain":
        params["T_ref"] = 0.2
    else:
        params["T_ref"] = 100.0
        
    params["phase"] == "all"
    # params["material_model"] = material_model
    params["active_model"] = active_model
    params["matparams_space"] = matparams_space

    solver_parameters, pressure, paramvec= make_solver_params(params, patient)


    Material = get_material_model(material_model)
    
    msg = "Should be {}, got {}".format(Material,
                                        type(solver_parameters["material"]))
    assert isinstance(solver_parameters["material"], Material), msg


    pressure 
    V_real = df.FunctionSpace(solver_parameters["mesh"],  "R", 0)
    gamma = df.Function(V_real, name = "gamma")
    

    matparams = setup_material_parameters(material_model)

    args = (patient.fiber,
            gamma,
            matparams,
            active_model,
            patient.sheet,
            patient.sheet_normal,
            params["T_ref"])

    
        

    material = Material(*args)

    solver_parameters["material"] = material
    solver = LVSolver(solver_parameters)
    solver.parameters["solve"]["snes_solver"]["report"] = True

    from pulse_adjoint.iterate import iterate

    pressures, volumes = [],[]
    # Increase pressure
    for plv in [0.0, 0.5, 1.0, 1.6]:
        iterate("pressure", solver, plv, pressure)
        u,p = solver.get_state().split(deepcopy=True)

        pressures.append(plv)
        
        vol = get_volume(patient, u = u)
        volumes.append(vol)

    
    # Increase gamma
    iterate("gamma", solver, 1.0, gamma)

    # 
    # df.plot(u, mode="displacement", interactive =True)
    

    
def test_holzapfel_ogden_active_strain():
    
    run("active_strain", "holzapfel_ogden", "R_0")

def test_neo_hookean_active_strain():
    
    run("active_strain", "neo_hookean", "R_0")


def test_holzapfel_ogden_active_stress():
    
    run("active_stress", "holzapfel_ogden", "R_0")

def test_neo_hookean_active_stress(): 
    
    run("active_stress", "neo_hookean", "R_0")


def test_guccione_active_stress(): 
    
    run("active_stress", "guccione", "R_0")


def test_guccione_active_strain(): 
    
    run("active_strain", "guccione", "R_0")

if __name__ == "__main__":
    test_neo_hookean_active_strain()
    test_holzapfel_ogden_active_strain()
    test_guccione_active_strain()

    
