"""
Inlate a geometry to a pressure using different 
material models, and compare with the Klotz curve.


"""
import dolfin as df
import numpy as np

from patient_data import LVTestPatient

from pulse_adjoint.setup_parameters import (setup_adjoint_contraction_parameters,
                                            setup_general_parameters)

from pulse_adjoint.setup_optimization import make_solver_params, get_material_model, get_volume

from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.material import HolzapfelOgden, Guccione, NeoHookean



setup_general_parameters()
patient = LVTestPatient()
patient.mesh.coordinates()[:] *= 3.15
ED_pressure = 1.6 #kPa


def setup_material_parameters(material_model):
    """
    Choose parameters based on 
    
    Hadjicharalambous, Myrianthi, et al. "Analysis of passive 
    cardiac constitutive laws for parameter estimation using 3D 
    tagged MRI." Biomechanics and modeling in mechanobiology 14.4 
    (2015): 807-828.
    

    These parameters did not really match the Klotz curve here.
    

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
    

def klotz_curve():
    """

    EDPVR based on Klotz curve

    Klotz, Stefan, et al. "Single-beat estimation of end-diastolic 
    pressure-volume relationship: a novel method with potential for 
    noninvasive application." American Journal of Physiology-Heart and 
    Circulatory Physiology 291.1 (2006): H403-H412.

    """
    

    
    import math
    
    # Some point at the EDPVR line
    Vm = 148.663 
    Pm = ED_pressure

    # Some constants
    An = 27.8 
    Bn = 2.76

    # kpa to mmhg
    Pm = Pm * 760/101.325

    
    V0 = Vm*(0.6 - 0.006*Pm)
    V30 = V0 + (Vm - V0) / (Pm/An)**(1.0/Bn)
    
    beta = math.log(Pm / 30.0) / math.log(Vm / V30)
    alpha = 30.0/V30**beta
    
    P_V0 = alpha*V0**beta

    vs = [V0]
    ps = [0.0]
    for p in np.linspace(1.0, 12.0):
        vi = (p/alpha)**(1.0/beta)
        vs.append(vi)
        ps.append(p * 101.325/760)

    return  vs, ps
   
   
    
    

def run_passive(active_model, material_model, matparams_space):

    params = setup_adjoint_contraction_parameters(material_model)
        
    params["active_model"] = active_model
    params["matparams_space"] = matparams_space
    params["base_bc"] = "fixed"

    solver_parameters, pressure, paramvec= make_solver_params(params, patient)


    Material = get_material_model(material_model)
    
    msg = "Should be {}, got {}".format(Material,
                                        type(solver_parameters["material"]))
    assert isinstance(solver_parameters["material"], Material), msg


    
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
    for plv in np.linspace(0,ED_pressure, 12):
        iterate("pressure", solver, plv, pressure)
        u,p = solver.get_state().split(deepcopy=True)

        pressures.append(plv)
        
        vol = get_volume(patient, u = u)
        volumes.append(vol)

    
    return volumes, pressures
    

def plot_passive_filling():
    
    neo_vols, neo_pres = run_passive("active_strain", "neo_hookean", "R_0")
    hol_vols, hol_pres = run_passive("active_strain", "holzapfel_ogden", "R_0")
    guc_vols, guc_pres = run_passive("active_strain", "guccione", "R_0")

 
    klotz_vol, klotz_pres = klotz_curve()

    import matplotlib.pyplot as plt
    from pulse_adjoint_post import plot
    plot.setup_plot()

    colors = plot.get_colormap(4)
    linestyles = ["-", "-.", "--"]
    

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(neo_vols, neo_pres, color = colors[0],
            linestyle = linestyles[0], label ="Neo Hookean")
    ax.plot(hol_vols, hol_pres, color = colors[1],
            linestyle = linestyles[1], label ="Holzapfel Ogden")
    ax.plot(guc_vols, guc_pres, color = colors[2],
            linestyle = linestyles[2], label ="Guccione")
    ax.plot( klotz_vol, klotz_pres, color = colors[3], label = "Klotz curve")

    ax.legend(loc="best")
    fig.savefig("passive_filling.png")
    plt.show()


if __name__ == "__main__":
    plot_passive_filling()
    
    
