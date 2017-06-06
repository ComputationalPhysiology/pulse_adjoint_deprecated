import os, yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    has_seaborn = True
except:
    has_seaborn = False
    pass


import dolfin

from pulse_adjoint.postprocess import load, utils
from pulse_adjoint.setup_optimization import (setup_adjoint_contraction_parameters,
                                              setup_general_parameters)

curdir = os.path.dirname(os.path.abspath(__file__))
setup_general_parameters()


dolfin.parameters["adjoint"]["stop_annotating"] = True


def plot_pv_loop(data, patient, params):

    simulated_volumes = utils.get_volumes(data["displacements"],
                                          patient,
                                          "lv",
                                          params["volume_approx"])
    

    

    fig = plt.figure()
    ax = fig.gca()
    
    ax.plot(patient.volume,patient.pressure, linestyle = "-",
            marker = "^", label = "measured")

    pressure = patient.pressure
    if params["unload"]:
        pressure = np.append(0, pressure)


    ax.plot(simulated_volumes, pressure,  linestyle = "-.",
            marker = "o",  label = "simulated")
    ax.legend(loc="best")
    plt.show()

def plot_strain(data, patient, params):

    simulated_strain = utils.get_regional_strains(data["displacements"],
                                                  patient,**params)


    region = 2
    
    fig = plt.figure()
    ax = fig.gca()

    c,r,l= np.transpose(patient.strain[region])
    ax.plot(c, linestyle = "-", marker = "^",label = "circumferential (measured)")
    ax.plot(r, linestyle = "-", marker = "^", label = "radial (measured)")
    ax.plot(l, linestyle = "-", marker = "^", label = "longitudinal (measured)")

    
    ax.plot(simulated_strain["circumferential"][region], marker = "o",
            linestyle = "-.",label = "circumferential (simulated)")
    ax.plot(simulated_strain["radial"][region], marker = "o",
            linestyle = "-.", label = "radial (simulated)")
    ax.plot(simulated_strain["longitudinal"][region], marker = "o",
            linestyle = "-.", label = "longitudinal (simulated)")

    
    ax.legend(loc="best")
    plt.show()

def make_visualization(data, patient, params, outdir_):

    features = {"displacement": data["displacements"],
                "gamma": data["gammas"]}

    outdir = "/".join([outdir_, "simulation"])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    utils.make_simulation(params, features, outdir, patient)

    


def main():

    result_dir = "results/simple_ellipsoid/"

    paramfile = "{}/input.yml".format(result_dir)
    params = load.load_parameters(paramfile)
    data, patient = load.get_data(params)
    
    plot_pv_loop(data, patient, params)
    plot_strain(data, patient, params)
    make_visualization(data, patient, params, result_dir)
    
  
if __name__ == "__main__":
    main()
