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
from dolfin import *
from campass.setup_optimization import setup_simulation, initialize_patient_data, setup_adjoint_contraction_parameters, setup_general_parameters
from campass.utils import  AutoVivification, Text,  pformat, passive_inflation_exists, contract_point_exists, get_spaces as get_strain_spaces
from campass.adjoint_contraction_args import *
import vtk, argparse, h5py, pickle, sys, collections, warnings, yaml
import numpy as np
import seaborn as sns
from copy import deepcopy
from tvtk.array_handler import *
import matplotlib as mpl
from matplotlib import pyplot as plt, rcParams, cbook, ticker, cm

ALL_ACTIVE_GROUP = "alpha_{}/reg_par_{}/active_contraction/contract_point_{}"
ALL_PASSIVE_GROUP = "alpha_{}/reg_par_0.0/passive_inflation"

setup_general_parameters()

STRAIN_REGION_NAMES = {1:"Anterior",
                       2:"Anteroseptal",
                       3:"Septum",
                       4:"Inferior",
                       5:"Posterior",
                       6:"Lateral",
                       7:"Anterior",
                       8:"Anteroseptal",
                       9:"Septum",
                       10:"Inferior",
                       11:"Posterior",
                       12:"Lateral",
                       13:"Anterior",
                       14:"Septum",
                       15:"Inferior",
                       16:"Lateral",
                       17:"Apex"}

STRAIN_REGIONS = {1:"LVBasalAnterior",
                 2:"LVBasalAnteroseptal",
                 3:"LVBasalSeptum",
                 4:"LVBasalInferior",
                 5:"LVBasalPosterior",
                 6:"LVBasalLateral",
                 7:"LVMidAnterior",
                 8:"LVMidAnteroseptal",
                 9:"LVMidSeptum",
                 10:"LVMidInferior",
                 11:"LVMidPosterior",
                 12:"LVMidLateral",
                 13:"LVApicalAnterior",
                 14:"LVApicalSeptum",
                 15:"LVApicalInferior",
                 16:"LVApicalLateral",
                 17:"LVApex"}

parameters["allow_extrapolation"] = True

# Plotting options
sns.set_palette("husl")
sns.set_style("white")
sns.set_style("ticks")
mpl.rcParams.update({'figure.autolayout': True})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 26} 

mpl.rc('font', **font)
mpl.pyplot.rc('text', usetex=True)
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

# Surpress warnings from matplotlib
warnings.filterwarnings("ignore", module="matplotlib")
    


####### Load stuff ################
def load_single_result(params, patient):


    if not os.path.isfile(params["sim_file"]):
        raise IOError("File {} does not exist".format(params["sim_file"]))

    a = params["alpha"]
    l = params["reg_par"]


    # Dictionaries to store data
    strain_dict = {strain : {i:[] for i in STRAIN_REGION_NUMS}  for strain in STRAIN_NUM_TO_KEY.values()}

    passive_data = {"misfit":{}, "material_parameters":{}, 
                    "strain": deepcopy(strain_dict), "states":[],
                    "gammas":[], "displacements":[],
                    "I1":[], "I4f":[], "p":[], "fiber_strains":[],
                    "strainfields":[], "fiber_stresses":[], "work":[], 
                    "fiber_work": [], "Eff":[], "Tff":[]}

    active_data = {a:{l:{"volume":[], "pressure":[], "gamma_gradient":[],
                   "strain": deepcopy(strain_dict),
                   "misfit":{"I_strain_initial":[], "I_volume_initial":[], 
                   "I_strain_optimal":[], "I_volume_optimal":[]},
                   "states":[], "gammas":[], "displacements":[],
                   "I1":[], "I4f":[], "p":[], "fiber_strains":[], "Eff":[], 
                   "strainfields":[], "fiber_stresses":[], "work":[],
                   "Tff":[], "fiber_work": []}}}
    
        
    # Groups within the result file
    passive_group = PASSIVE_INFLATION_GROUP.format(params["alpha_matparams"])
    passive_misfit_str = "/".join([passive_group, "misfit/misfit_functional/{}/{}"])
    passive_volume_str = "/".join([passive_group, "volume"])
    passive_pressure_str = "/".join([passive_group, "lv_pressures"])
    passive_strain_str = "/".join([passive_group, "strains/{}/region_{}"])
    passive_matparams_str = "/".join([passive_group, "parameters/{}_material_parameters"])

    main_active_group = "/".join(ACTIVE_CONTRACTION_GROUP.split("/")[:-1]).format(params["alpha"])


    with h5py.File(params["sim_file"] , "r") as h5file:
        
        
        ## Passive inflation
        if passive_group in h5file:

            # Volume
            passive_data["volume"] = np.array(h5file[passive_volume_str])

            # Pressure
            passive_data["pressure"] = np.array(h5file[passive_pressure_str])

            # Strain
            for point in range(patient.passive_filling_duration):
                for region in STRAIN_REGION_NUMS:
                    strain_arr = np.array(h5file[passive_strain_str.format(point,region)])
                    for direction in STRAIN_NUM_TO_KEY.keys():
                        passive_data["strain"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr[direction])

            
            for s in ["optimal", "initial"]:

                # Material parameters
                passive_data["material_parameters"][s] = np.array(h5file[passive_matparams_str.format(s)])
                get_matparams = False


                # Misfit
                passive_data["misfit"]["I_strain_{}".format(s)] = \
                  np.array(h5file[passive_misfit_str.format(s, "strain")])[0]
                passive_data["misfit"]["I_volume_{}".format(s)] = \
                  np.array(h5file[passive_misfit_str.format(s, "volume")])[0]
                  
        else:
            passive_data = None
            pass
            print "Warning: passive group {} does not exist".format(passive_group)

        p = 0
        if main_active_group in h5file:
            
                
            while "contract_point_{}".format(p) in h5file[main_active_group].keys():

                active_group = ACTIVE_CONTRACTION_GROUP.format(params["alpha"], p)
                active_misfit_str = "/".join([active_group, "misfit/misfit_functional/{}/{}"])
                active_grad_str = "/".join([active_group, "parameters/activation_parameter_gradient_size"])
                active_volume_str = "/".join([active_group, "volume"])
                active_pressure_str = "/".join([active_group, "lv_pressures"])
                active_strain_str = "/".join([active_group, "strains/0/region_{}"])


                # Volume
                active_data[a][l]["volume"].append(np.array(h5file[active_volume_str])[0])

                # Pressure
                active_data[a][l]["pressure"].append(np.array(h5file[active_pressure_str])[0])
                    

                # Strain
                for region in STRAIN_REGION_NUMS:
                    strain_arr = np.array(h5file[active_strain_str.format(region)])
                        
                    for direction in STRAIN_NUM_TO_KEY.keys():
                        active_data[a][l]["strain"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr[direction])

                # Gamma gradient
                active_data[a][l]["gamma_gradient"].append(np.array(h5file[active_grad_str])[0])

                # Misfit
                for s in ["optimal", "initial"]:
                    active_data[a][l]["misfit"]["I_strain_{}".format(s)].append(
                        np.array(h5file[active_misfit_str.format(s, "strain")])[0])
                    active_data[a][l]["misfit"]["I_volume_{}".format(s)].append(
                          np.array(h5file[active_misfit_str.format(s, "volume")])[0])


                p += 1
                    
                
            active_data[a][l]["num_points"] = p-1
            
    measured_data = load_measured_strain_and_volume(patient)
    data = {"passive":passive_data, "active":active_data, "measured":measured_data}

    # Done loading data using h5py. Now load dolfin data
    kwargs = init_spaces(patient.mesh, params["gamma_space"])
    
    # Some indices
    passive_filling_duration = patient.passive_filling_duration
    num_contract_points = patient.num_contract_points
    # Total points 
    num_points = num_contract_points + passive_filling_duration-1
    
    # Time stamps
    time_stamps = np.subtract(patient.time,patient.time[0])
    time_stamps = time_stamps[:num_points]
    


    opt_keys = ["nfev", "njev", "nit", "run_time", "ncrash", "message"]

    gamma = Function(kwargs["gamma_space"], name = "Contraction Parameter")
    state = Function(kwargs["state_space"], name = "State")
    strainfield = Function(kwargs["strainfield_space"], name = "Strainfield")
    

    # Setup a solver for stress and work calculations
    p_lv = Expression("t", t = 0.0)
    solver = get_solver(params, data, patient, gamma, p_lv, patient.strain_markers)


    with HDF5File(mpi_comm_world(), params["sim_file"], "r") as h5file:

        print "Reading passive data"

        passive_data["passive_attrs"] = {}
        # Load passive data

        for key in opt_keys:
            passive_data["passive_attrs"][key] = h5file.attributes(passive_group)[key]

        passive_data["num_points"] = passive_filling_duration

        for pv_num in range(passive_filling_duration):
            
            h5file.read(state, passive_group + "/states/{}".format(pv_num))

            u,p = state.split(deepcopy=True)
            passive_data["displacements"].append(Vector(u.vector()))
            passive_data["states"].append(Vector(state.vector()))
            passive_data["gammas"].append(Vector(gamma.vector()))
            passive_data["p"].append(Vector(p.vector()))

            p_lv.t = data["passive"]["pressure"][pv_num]
            get_mechanical_features(solver, state, gamma, kwargs, passive_data)


            try:
                h5file.read(strainfield, h5group + "/strainfields/{}".format(pv_num))
                passive_data["strainfields"].append(Vector(strainfield.vector()))
            except: pass

        data["passive"].update(**passive_data)


        # Load active data
        print "Reading active data"
        
        n = data["active"][a][l]["num_points"]
        active_data[a][l]["active_attrs"] = {key:[] for key in opt_keys}


        for pv_num in range(n+1):

            active_group = ACTIVE_CONTRACTION_GROUP.format(params["alpha"], pv_num)

            for key in opt_keys:
                active_data[a][l]["active_attrs"][key].append(h5file.attributes(active_group)[key])

                    
            h5file.read(state, active_group + "/states/0")
            h5file.read(gamma, active_group + "/parameters/activation_parameter_function")
            u,p = state.split(deepcopy=True)

            active_data[a][l]["displacements"].append(Vector(u.vector()))
            active_data[a][l]["states"].append(Vector(state.vector()))
            active_data[a][l]["gammas"].append(Vector(gamma.vector()))
            active_data[a][l]["p"].append(Vector(p.vector()))

                    
            p_lv.t = data["active"][a][l]["pressure"][pv_num]
            get_mechanical_features(solver, state, gamma, kwargs,active_data[a][l])


            try:
                h5file.read(strainfield, active_group + "/strainfields/0")
                active_data[a][l]["strainfields"].append(Vector(strainfield.vector()))
            except: pass

        data["active"][a][l].update(**active_data[a][l])

    

    # Add some extra stuff to arguments
    kwargs["mesh"] = patient.mesh
    kwargs["strain_markers"] = patient.strain_markers
    kwargs["time_stamps"] = time_stamps
    kwargs["dx"] = Measure("dx", subdomain_data = patient.strain_markers,
                           domain = patient.mesh)
    kwargs["num_points"] = num_points
    kwargs["passive_filling_begins"] = patient.passive_filling_begins
    kwargs["passive_filling_duration"] = passive_filling_duration
    kwargs["num_contract_points"] = num_contract_points
    
    return data, kwargs


def load_geometry_and_microstructure_from_results(patient, params):

    from mesh_generation.mesh_utils import load_geometry_from_h5
    geo = load_geometry_from_h5(params["sim_file"])
    # from IPython import embed; embed()
    # exit()
    

def get_h5py_data(params, patient, alpha_regpars, synthetic_data=False):
    """
    Get misfit, volumes, pressures and material parameters

    """

    strain_dict = {strain : {i:[] for i in STRAIN_REGION_NUMS}  for strain in STRAIN_NUM_TO_KEY.values()}

    passive_data = {"misfit":{}, "material_parameters":{}, 
                    "strain": deepcopy(strain_dict)}

    
    active_data = AutoVivification()
    for a, l in alpha_regpars:
        active_data[a][l] = {"volume":[], "pressure":[], "gamma_gradient":[], 
                             "misfit":{"I_strain_initial":[], "I_volume_initial":[], 
                                       "I_strain_optimal":[], "I_volume_optimal":[]} }
        active_data[a][l]["strain"] = deepcopy(strain_dict)

    
    
    # Passive group names
    passive_group = ALL_PASSIVE_GROUP.format(params["alpha_matparams"])
    # passive_group = 'alpha_{}/passive_inflation'.format(params["alpha_matparams"])
    passive_misfit_str = "/".join([passive_group, "misfit/misfit_functional/{}/{}"])
    passive_volume_str = "/".join([passive_group, "volume"])
    passive_pressure_str = "/".join([passive_group, "lv_pressures"])
    passive_strain_str = "/".join([passive_group, "strains/{}/region_{}"])
    passive_matparams_str = "/".join([passive_group, "parameters/{}_material_parameters"])

    get_matparams = True
    matparams = None
    

    if not os.path.isfile(params["sim_file"]):
        raise IOError("File {} does not exist".format(sim_file))

    with h5py.File(params["sim_file"] , "r") as h5file:
        
        
        ## Passive inflation
        if passive_group in h5file:

            # Volume
            passive_data["volume"] = np.array(h5file[passive_volume_str])

            # Pressure
            passive_data["pressure"] = np.array(h5file[passive_pressure_str])

            # Strain
            for point in range(patient.passive_filling_duration):
                for region in STRAIN_REGION_NUMS:
                    strain_arr = np.array(h5file[passive_strain_str.format(point,region)])
                    for direction in STRAIN_NUM_TO_KEY.keys():
                        passive_data["strain"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr[direction])

            
            for s in ["optimal", "initial"]:

                # Material parameters
                passive_data["material_parameters"][s] = np.array(h5file[passive_matparams_str.format(s)])
                get_matparams = False


                # Misfit
                passive_data["misfit"]["I_strain_{}".format(s)] = \
                  np.array(h5file[passive_misfit_str.format(s, "strain")])[0]
                passive_data["misfit"]["I_volume_{}".format(s)] = \
                  np.array(h5file[passive_misfit_str.format(s, "volume")])[0]
                  
        else:
            passive_data = None
            pass
            print "Warning: passive group {} does not exist".format(passive_group)
        
        ## Active contraction
        print "alpha\tlambda\tnum_contract_points"
        for alpha, reg_par in alpha_regpars:
            
            active_group = "alpha_{}/reg_par_{}/active_contraction/".format(alpha, reg_par)
            p = 0
            if active_group in h5file:

                
                while "contract_point_{}".format(p) in \
                  h5file[active_group].keys():


                    # Active group names
                    active_misfit_str = "/".join([ALL_ACTIVE_GROUP.format(alpha, reg_par, p), 
                                                  "misfit/misfit_functional/{}/{}"])
                    active_grad_str = "/".join([ALL_ACTIVE_GROUP.format(alpha, reg_par, p),
                                                "parameters/activation_parameter_gradient_size"])
                    active_volume_str = "/".join([ALL_ACTIVE_GROUP.format(alpha, reg_par, p), "volume"])
                    active_pressure_str = "/".join([ALL_ACTIVE_GROUP.format(alpha, reg_par, p), "lv_pressures"])
                    active_strain_str = "/".join([ALL_ACTIVE_GROUP.format(alpha, reg_par, p), 
                                                  "strains/0/region_{}"])


                    # Volume
                    active_data[alpha][reg_par]["volume"].append(np.array(h5file[active_volume_str])[0])

                    # Pressure
                    active_data[alpha][reg_par]["pressure"].append(np.array(h5file[active_pressure_str])[0])
                    

                    # Strain
                    for region in STRAIN_REGION_NUMS:
                        strain_arr = np.array(h5file[active_strain_str.format(region)])
                        
                        for direction in STRAIN_NUM_TO_KEY.keys():
                            active_data[alpha][reg_par]["strain"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr[direction])

                    # Gamma gradient
                    active_data[alpha][reg_par]["gamma_gradient"].append(np.array(h5file[active_grad_str])[0])

                    # Misfit
                    for s in ["optimal", "initial"]:
                        active_data[alpha][reg_par]["misfit"]["I_strain_{}".format(s)].append(
                          np.array(h5file[active_misfit_str.format(s, "strain")])[0])
                        active_data[alpha][reg_par]["misfit"]["I_volume_{}".format(s)].append(
                          np.array(h5file[active_misfit_str.format(s, "volume")])[0])

                    
                    # FIXME: SHOULD STORE THE MATERIAL PARAMETERS AS PART OF THE SYNTHETIC DATA
                    if get_matparams:
                        with HDF5File(mpi_comm_world(), params["sim_file"], "r") as h5file2:
                            # We need to load the material parameters used
                            a = h5file2.attributes(ALL_ACTIVE_GROUP.format(alpha, reg_par, 0))["Material_parameters/a"]
                            a_f = h5file2.attributes(ALL_ACTIVE_GROUP.format(alpha, reg_par, 0))["Material_parameters/a_f"]
                            b = h5file2.attributes(ALL_ACTIVE_GROUP.format(alpha, reg_par, 0))["Material_parameters/b"]
                            b_f = h5file2.attributes(ALL_ACTIVE_GROUP.format(alpha, reg_par, 0))["Material_parameters/b_f"]
                            matparams = np.array([a, a_f, b, b_f])

                            get_matparams = False

                    p += 1
                    
                
                active_data[alpha][reg_par]["num_points"] = p-1

            else:
                active_data[alpha][reg_par] = None

            print "{}\t{:.1e}\t{}".format(alpha, reg_par, p-1)
            # s = "optimal"
            # print "I_strain", active_data[alpha][reg_par]["misfit"]["I_strain_{}".format(s)]
            # print "I_volume", active_data[alpha][reg_par]["misfit"]["I_volume_{}".format(s)]


        if synthetic_data:
            # Load synthetic data

            num_points = patient.passive_filling_duration \
              + patient.num_contract_points 

            noise = params["noise"]
            
            synth_data = {"volume":[], "pressure":[]}
            synth_data["strain"] = deepcopy(strain_dict)
            synth_data["strain_original"] = deepcopy(strain_dict) if noise else None
            synth_data["strain_w_noise"] = deepcopy(strain_dict) if noise else None
            synth_data["strain_corrected"] = deepcopy(strain_dict) if noise else None

            synth_str = "synthetic_data/point_{}"            

            for point in range(num_points):
                h5group = synth_str.format(point)
                # Get strains
                for region in STRAIN_REGION_NUMS:
                    strain_arr = np.array(h5file[h5group + "/strain/region_{}".format(region)])
                    if noise:
                        strain_arr_noise = np.array(h5file[h5group + "/strain_w_noise/region_{}".format(region)])
                        strain_arr_orig = np.array(h5file[h5group + "/original_strain/region_{}".format(region)])
                        strain_arr_corr = np.array(h5file[h5group + "/corrected_strain/region_{}".format(region)])

                    for direction in STRAIN_NUM_TO_KEY.keys():
                        synth_data["strain"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr[direction])
                        if noise:
                            synth_data["strain_w_noise"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr_noise[direction])
                            synth_data["strain_original"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr_orig[direction])
                            synth_data["strain_corrected"][STRAIN_NUM_TO_KEY[direction]][region].append(strain_arr_corr[direction])


                # Get volume
                v = np.array(h5file[h5group + "/volume"])
                synth_data["volume"].append(v[0])

                # Get pressure
                p = np.array(h5file[h5group + "/pressure"])
                synth_data["pressure"].append(p[0]*10) # kPa
            
            # REMOVE THIS IF NOT PASSIVE PHASE IS EQUAL TO SYNTHETIC
            # Update passive data
            if passive_data is None:
                passive_data = {"misfit":{}, "material_parameters":{},  "strain": deepcopy(strain_dict)}

                passive_data["volume"] = synth_data["volume"][:patient.passive_filling_duration]
                passive_data["pressure"] = synth_data["pressure"][:patient.passive_filling_duration]

                passive_strain = {}
                for s in STRAIN_NUM_TO_KEY.values():
                    passive_strain[s] = {}
                    for i in STRAIN_REGION_NUMS:
                        if noise:
                            passive_strain[s][i] = synth_data["strain_original"][s][i][:patient.passive_filling_duration]
                        else:
                            passive_strain[s][i] = synth_data["strain"][s][i][:patient.passive_filling_duration]
                passive_data["strain"] = passive_strain

                for s in ["optimal", "initial"]:

                    # Material PARAMETERS
                    passive_data["material_parameters"][s] = matparams

                    # Misfit
                    passive_data["misfit"]["I_strain_{}".format(s)] = 0.0
                    passive_data["misfit"]["I_volume_{}".format(s)] = 0.0

            return {"passive":passive_data, "active":active_data, "synthetic":synth_data}
        else:
            # Get measured data
            measured_data = load_measured_strain_and_volume(patient)

            return {"passive":passive_data, "active":active_data, "measured":measured_data, "synthetic":None}
            

def get_dolfin_data(params, patient, alpha_regpars, data):
    
    kwargs = init_spaces(patient.mesh, params["gamma_space"])
    
    # Some indices
    passive_filling_duration = patient.passive_filling_duration
    num_contract_points = patient.num_contract_points
    # Total points 
    num_points = num_contract_points + passive_filling_duration-1
    
    # Time stamps
    time_stamps = np.subtract(patient.time,patient.time[0])
    time_stamps = time_stamps[:num_points]

    passive_data = {"states":[], "gammas":[], "displacements":[],
                    "I1":[], "I4f":[], "p":[], "fiber_strains":[],
                    "strainfields":[], "fiber_stresses":[], "work":[], 
                    "fiber_work": [], "Eff":[], "Tff":[]}
    


    opt_keys = ["nfev", "njev", "nit", "run_time", "ncrash", "message"]

    gamma = Function(kwargs["gamma_space"], name = "Contraction Parameter")
    state = Function(kwargs["state_space"], name = "State")
    strainfield = Function(kwargs["strainfield_space"], name = "Strainfield")
    

    # Setup a solver for stress and work calculations
    p_lv = Expression("t", t = 0.0)
    solver = get_solver(params, data, patient, gamma, p_lv, patient.strain_markers)
  

    with HDF5File(mpi_comm_world(), params["sim_file"], "r") as h5file:

        print "Reading passive data"

        passive_data["passive_attrs"] = {}
        # Load passive data
        h5group = ALL_PASSIVE_GROUP.format(params["alpha_matparams"])
        # h5group = 'alpha_{}/passive_inflation'.format(params["alpha_matparams"])

        for key in opt_keys:
            passive_data["passive_attrs"][key] = h5file.attributes(h5group)[key]

        passive_data["num_points"] = passive_filling_duration

        for pv_num in range(passive_filling_duration):
            
            h5file.read(state, h5group + "/states/{}".format(pv_num))

            u,p = state.split(deepcopy=True)
            passive_data["displacements"].append(Vector(u.vector()))
            passive_data["states"].append(Vector(state.vector()))
            passive_data["gammas"].append(Vector(gamma.vector()))
            passive_data["p"].append(Vector(p.vector()))

            p_lv.t = data["passive"]["pressure"][pv_num]
            get_mechanical_features(solver, state, gamma, kwargs, passive_data)


            try:
                h5file.read(strainfield, h5group + "/strainfields/{}".format(pv_num))
                passive_data["strainfields"].append(Vector(strainfield.vector()))
            except: pass

        data["passive"].update(**passive_data)


        # Load active data
        print "Reading active data"
        print "alpha\tlambda"
        for alpha, reg_par in alpha_regpars:
            if data["active"][alpha][reg_par]:
                print "{}\t{:.2e}".format(alpha, reg_par)
                n = data["active"][alpha][reg_par]["num_points"]


                active_data = {"states":[], "gammas":[], "displacements":[],
                               "I1":[], "I4f":[], "p":[], "fiber_strains":[], "Eff":[], 
                               "strainfields":[], "fiber_stresses":[], "work":[], "Tff":[], 
                               "fiber_work": [], "active_attrs":{key:[] for key in opt_keys}}

                for pv_num in range(n+1):

                    h5group = ALL_ACTIVE_GROUP.format(alpha, reg_par, pv_num)

                    for key in opt_keys:
                        active_data["active_attrs"][key].append(h5file.attributes(h5group)[key])

                    
                    h5file.read(state, h5group + "/states/0")
                    h5file.read(gamma, h5group + "/parameters/activation_parameter_function")
                    u,p = state.split(deepcopy=True)

                    active_data["displacements"].append(Vector(u.vector()))
                    active_data["states"].append(Vector(state.vector()))
                    active_data["gammas"].append(Vector(gamma.vector()))
                    active_data["p"].append(Vector(p.vector()))

                    
                    p_lv.t = data["active"][alpha][reg_par]["pressure"][pv_num]
                    get_mechanical_features(solver, state, gamma, kwargs,active_data)


                    try:
                        h5file.read(strainfield, h5group + "/strainfields/0")
                        active_data["strainfields"].append(Vector(strainfield.vector()))
                    except: pass

                data["active"][alpha][reg_par].update(**active_data)

    

    # Add some extra stuff to arguments
    kwargs["mesh"] = patient.mesh
    kwargs["strain_markers"] = patient.strain_markers
    kwargs["time_stamps"] = time_stamps
    kwargs["dx"] = Measure("dx", subdomain_data = patient.strain_markers,
                           domain = patient.mesh)
    kwargs["num_points"] = num_points
    kwargs["passive_filling_begins"] = patient.passive_filling_begins
    kwargs["passive_filling_duration"] = passive_filling_duration
    kwargs["num_contract_points"] = num_contract_points

    return  data, kwargs
    

def get_dolfin_data_synthetic(params, patient, alpha_regpars, data):
    
    kwargs = init_spaces(patient.mesh, params["gamma_space"])
    
    # Some indices
    passive_filling_duration = patient.passive_filling_duration
    num_contract_points = patient.num_contract_points
    # Total points 
    num_points = num_contract_points + passive_filling_duration-1
    
    # Time stamps
    time_stamps = np.subtract(patient.time,patient.time[0])
    time_stamps = time_stamps[:num_points]

    passive_data = {"states":[], "gammas":[], "displacements":[],
                    "I1":[], "I4_f":[], "p":[], "Eff":[], "Tff": [],
                    "strainfields":[], "stresses":[], "work":[] }
    

    
    # Setup a solver for stress and work calculations
    solver = get_solver(params, data, patient)
  

    opt_keys = ["nfev", "njev", "nit", "run_time", "ncrash", "message"]

    gamma = Function(kwargs["gamma_space"], name = "Contraction Parameter")
    state = Function(kwargs["state_space"], name = "State")
    strainfield = Function(kwargs["strainfield_space"], name = "Strainfield")
    stress = Function(kwargs["gamma_space"], name = "Contraction Parameter")
    Eff = Function(kwargs["quad_space"], name = "Eff")
    Tff = Function(kwargs["quad_space"], name = "Tff")

    with HDF5File(mpi_comm_world(), params["sim_file"], "r") as h5file:

        # Load active data
        print "Reading active data"
        print "alpha\tlambda"
        for alpha, reg_par in alpha_regpars:
            if data["active"][alpha][reg_par]:
                print "{}\t{:.2e}".format(alpha, reg_par)
                n = data["active"][alpha][reg_par]["num_points"]


                active_data = {"states":[], "gammas":[], "displacements":[],
                               "I1":[], "I4_f":[], "p":[], "Eff":[], "Tff": [],
                               "stresses":[], "strainfields":[], "work":[], 
                               "active_attrs":{key:[] for key in opt_keys}}
            
                for pv_num in range(n+1):

                    h5group = ALL_ACTIVE_GROUP.format(alpha, reg_par, pv_num)

                    for key in opt_keys:
                        active_data["active_attrs"][key].append(h5file.attributes(h5group)[key])

                    
                    h5file.read(state, h5group + "/states/0")
                    h5file.read(gamma, h5group + "/parameters/activation_parameter_function")
                    u,p = state.split(deepcopy=True)

                    active_data["displacements"].append(Vector(u.vector()))
                    active_data["states"].append(Vector(state.vector()))
                    active_data["gammas"].append(Vector(gamma.vector()))

                    I1, I4_f, Eff, Tff  = get_mechanical_features(solver, state)
                    active_data["I1"].append(Vector(I1.vector()))
                    active_data["I4_f"].append(Vector(I4_f.vector()))
                    active_data["p"].append(Vector(p.vector()))

                    stress = compute_stress(solver, state, kwargs["stress_space"], patient.e_f)
                    active_data["stresses"].append(Vector(stress.vector()))
                    

                    work = compute_cardiac_work(solver, state, kwargs["stress_space"], patient.e_f)
                    active_data["work"].append(Vector(work.vector()))

                    try:
                        h5file.read(strainfield, h5group + "/strainfields/0")
                        active_data["strainfields"].append(Vector(strainfield.vector()))
                    except: pass

                data["active"][alpha][reg_par].update(**active_data)

                


        
        # Check if we have synthetic data
        if data["synthetic"]:
            
            num_points = patient.passive_filling_duration \
              + patient.num_contract_points 

            synth_data = {"states":[], "gammas":[], "work":[],
                          "I1":[], "I4_f":[], "p":[],
                          "displacements":[], "stresses":[]}


            synth_str = "synthetic_data/point_{}"
            
            for point in range(num_points):
               
                h5group = synth_str.format(point)

                # Get state
                h5file.read(state, h5group + "/state")
                synth_data["states"].append(Vector(state.vector()))

                # Get displacement
                u,p = state.split(deepcopy=True)
                synth_data["displacements"].append(Vector(u.vector()))

                # Get gamma
                h5file.read(gamma, h5group + "/activation_parameter")
                synth_data["gammas"].append(Vector(gamma.vector()))

                # Mechanical features
                I1, I4_f,  Eff, Tff = get_mechanical_features(solver, state)
                synth_data["I1"].append(Vector(I1.vector()))
                synth_data["I4_f"].append(Vector(I4_f.vector()))
                synth_data["p"].append(Vector(p.vector()))

                # Copmute stress
                stress = compute_stress(solver, state, kwargs["stress_space"], patient.e_f)
                synth_data["stresses"].append(Vector(stress.vector()))
                

                # Copmute cardiac work
                work = compute_cardiac_work(solver, state, kwargs["stress_space"], patient.e_f)
                synth_data["work"].append(Vector(work.vector()))

                # The passive data is the same as synthetic data
                if point < patient.passive_filling_duration:
            
                    passive_data["states"].append(Vector(state.vector()))
                    passive_data["displacements"].append(Vector(u.vector()))
                    passive_data["stresses"].append(Vector(stress.vector()))
                    passive_data["gammas"].append(Vector(gamma.vector()))
                    passive_data["work"].append(Vector(work.vector()))
                    passive_data["I1"].append(Vector(I1.vector()))
                    passive_data["I4_f"].append(Vector(I4_f.vector()))
                    passive_data["p"].append(Vector(p.vector()))

            data["synthetic"].update(**synth_data)
            data["passive"].update(**passive_data)

        # Check that the passive group exist
        # if data["passive"]:
        else:
            print "Reading passive data"

            passive_data["passive_attrs"] = {}
            # Load passive data
            h5group = ALL_PASSIVE_GROUP.format(params["alpha_matparams"])
            # h5group = 'alpha_{}/passive_inflation'.format(params["alpha_matparams"])

            for key in opt_keys:
                passive_data["passive_attrs"][key] = h5file.attributes(h5group)[key]

            passive_data["num_points"] = passive_filling_duration

            for pv_num in range(passive_filling_duration):

                h5file.read(state, h5group + "/states/{}".format(pv_num))
                
                u,p = state.split(deepcopy=True)
                passive_data["displacements"].append(Vector(u.vector()))
                passive_data["states"].append(Vector(state.vector()))
                passive_data["gammas"].append(Vector(gamma.vector()))

                I1, I4_f, Eff, Tff  = get_mechanical_features(solver, state)
                passive_data["I1"].append(Vector(I1.vector()))
                passive_data["I4_f"].append(Vector(I4_f.vector()))
                passive_data["p"].append(Vector(p.vector()))
                
                stress = compute_stress(solver, state, kwargs["stress_space"], patient.e_f)
                passive_data["stresses"].append(Vector(stress.vector()))

                work = compute_cardiac_work(solver, state, kwargs["stress_space"], patient.e_f)
                passive_data["work"].append(Vector(work.vector()))
                
                try:
                    h5file.read(strainfield, h5group + "/strainfields/{}".format(pv_num))
                    passive_data["strainfields"].append(Vector(strainfield.vector()))
                except: pass
                
            data["passive"].update(**passive_data)

        
    

    # Add some extra stuff to arguments
    kwargs["mesh"] = patient.mesh
    kwargs["strain_markers"] = patient.strain_markers
    kwargs["time_stamps"] = time_stamps
    kwargs["dx"] = Measure("dx", subdomain_data = patient.strain_markers,
                           domain = patient.mesh)
    kwargs["num_points"] = num_points
    kwargs["passive_filling_duration"] = passive_filling_duration
    kwargs["num_contract_points"] = num_contract_points

    return  data, kwargs


def get_all_data(params, patient, alpha_regpars, synthetic_data = False):
    
    print Text.blue("\nLoad simulated data")
    
    print Text.yellow("\nReading h5py data")
    h5py_data = get_h5py_data(params, patient, alpha_regpars, synthetic_data)
    print "Done reading h5py data"

    print Text.yellow("\nReading dolfin data...")
    data, kwargs = get_dolfin_data(params, patient, alpha_regpars, h5py_data)
    print "Done reading dolfin data"
    
    print "Remove null data"
    clean_up_dict(data)
    
    
    return data, kwargs


def load_measured_strain_and_volume(patient, num_points = None):

    num_points = patient.num_points if num_points is None else num_points

    # Volume
    cav_volume = compute_inner_cavity_volume(patient.mesh, patient.facets_markers, patient.ENDO)
    volume_offset = patient.volume[0] - cav_volume
    volumes = np.subtract(patient.volume[:num_points+1], volume_offset)

    # Strain
    strains = {strain : {i:[] for i in STRAIN_REGION_NUMS} for strain in STRAIN_NUM_TO_KEY.values()}
    for region in STRAIN_REGION_NUMS:
        for direction in range(3):
            regional_strain = patient.strain[region]
            strains[STRAIN_NUM_TO_KEY[direction]][region] = np.array([s[direction] for s in regional_strain[:(num_points+1)]])


    # Pressure
    pressure = np.array(patient.pressure[:num_points+1])
    # pressure *= KPA_TO_CPA
    pressure_offset = pressure[0]
    pressure = np.subtract(pressure, pressure_offset)
   
       

    data = {}
    data["volume"] = volumes
    data["strain"] = strains
    data["pressure"] = pressure
    return data

def get_strain_partly(strain, n):
    part_strain = {}
    for s in STRAIN_NUM_TO_KEY.values():
        part_strain[s] = {}
        for i in STRAIN_REGION_NUMS:
                part_strain[s][i] = strain[s][i][:n]
            
    return part_strain


########## Save stuff #################
def save_misfit_functionals_to_pickle(params, res_dir):
    err_data = {}
    alpha = params["alpha"]
    reg_par = params["reg_par"]

    print "alpha = {}, reg_par = {}".format(alpha, reg_par)
    res_path = res_dir + "/results_{}_{}.p".format(alpha,reg_par)
    try:
        with open(res_path, "rb") as f:
            err_data = pickle.load(f)
    except:
        err_data = {}

    err_data["misfit"] = get_mistfit_functionals(params["sim_file"], 
                                                 params["alpha"],
                                                 params["alpha_matparams"])

    res_path = res_dir + "/results_{}_{}.p".format(alpha, reg_par)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(res_path, "wb" ) as output:
        pickle.dump(err_data, output, pickle.HIGHEST_PROTOCOL)


def update_dict(d, u):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
def save_dict_to_pickle(data, outfile):
    with open(outfile, "wb" ) as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

def save_results_to_h5(sim_data, spaces, h5name):

    of = "a" if os.path.exists(h5name) else "w"

    args = spaces["args"]
    h5key = "/".join(["gamma_space_"+args.gamma_space, 
                    "alpha_" + args.alpha, 
                    "reg_par_"+args.reg_par])

    if of == "a":
        with h5py.File(h5name) as h5pyfile:
            if h5key in h5pyfile:
                del h5pyfile[h5key]


    keys = ['gammas', 'states', 'displacements']
    assert set(keys) < set(sim_data.keys())

    with HDF5File(mpi_comm_world(), h5name, of) as h5file:
        h5file.write(spaces["mesh"], "mesh")
        h5file.attributes("mesh")["displacement_space"] = "CG_2"
        h5file.attributes("mesh")["state_space"] = "CG_2xCG_1"
        h5file.attributes("mesh")["num_points"] = len(spaces["time_stamps"])
   

        for k in keys:
            for i,f in enumerate(sim_data[k]):
                h5file.write(f, "/".join([h5key, k, "point_"+str(i)]))



####### Create/set up stuff #########
def get_solver(params, data, patient, gamma, p_lv, strain_markers = None):

    from campass.setup_optimization import setup_solver_parameters
    from haosolver import Compressibility, ActiveHaoSolver

    def make_dirichlet_bcs(W):
	'''Make Dirichlet boundary conditions where the base is allowed to slide
        in the x = 0 plane.
        '''
        no_base_x_tran_bc = DirichletBC(W.sub(0).sub(0), 0, patient.BASE)
        return [no_base_x_tran_bc]

    
    from campass.material import HolzapfelOgden

    matparams_arr = data["passive"]["material_parameters"]["optimal"]
    matparams = {"a": matparams_arr[0], 
                 "a_f": matparams_arr[1],
                 "b": matparams_arr[2], 
                 "b_f": matparams_arr[3]}

    material = HolzapfelOgden(f0 = patient.e_f, 
                              gamma = gamma, 
                              params = matparams,
                              active_model = "active_stress", 
                              strain_markers = strain_markers)

    

    solver_parameters = {"mesh": patient.mesh,
                         "facet_function": patient.facets_markers,
                         "facet_normal": FacetNormal(patient.mesh),
                         "state_space": "P_2:P_1",
                         "compressibility":{"type": "incompressible",
                                            "lambda":0.0},
                         "material": material,
                         "bc":{"dirichlet": make_dirichlet_bcs,
                               "Pressure":[[p_lv, patient.ENDO]],
                               "Robin":[[-Constant(params["base_spring_k"], 
                                                   name ="base_spring_constant"), patient.BASE]]},
                         "solve":setup_solver_parameters()}

    from campass.lvsolver import LVSolver

    return LVSolver(solver_parameters)


    
def merge_passive_active(data, alpha, reg_par, key, return_n = False):
    """
    Merge passive and active data for a given key
    Possible keys might be:
      volume, strain, gammas, displacements, states, pressures, strainfields
      
    """
    
    assert key in data["passive"].keys(), "{} not passive data".format(key)
    msg="{} not active data with alpha = {}, reg_par = {}".format(key, alpha, reg_par)
    assert key in data["active"][alpha][reg_par].keys(), msg
    

    passive_data = data["passive"][key]
    active_data = data["active"][alpha][reg_par][key]

    if key == "strain":
        pas_act = {}
        for d in passive_data.keys():
            pas_act[d] = {}
            for r in passive_data[d].keys():
                pas_act[d][r] = passive_data[d][r] + active_data[d][r]
                n = len(pas_act[d][r])

        if return_n:
            return pas_act, n 

        return pas_act

    else:
        
        if isinstance(passive_data, np.ndarray): passive_data = passive_data.tolist()
        if isinstance(active_data, np.ndarray): active_data = active_data.tolist()

        pas_act = np.array(passive_data + active_data)
        if return_n:
            return pas_act, len(pas_act)
        else:
            return pas_act

def clean_up_dict(data):
    """
    Remove keys with no data
    """
    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():
    
            if not data["active"][alpha][reg_par]:
                data["active"][alpha].pop(reg_par, None)


    for alpha in data["active"].keys():

        if not data["active"][alpha]:
            data["active"].pop(alpha, None)

def init_spaces(mesh, gamma_space = "CG_1"):
    
    spaces = {}
    # spaces["strain_space"] =  VectorFunctionSpace(mesh, "R", 0, dim = 3)
    # spaces["real_space"] = FunctionSpace(mesh, "R", 0)
    spaces["marker_space"] = FunctionSpace(mesh, "DG", 0)
    spaces["stress_space"] = FunctionSpace(mesh, "CG", 1)

    if gamma_space == "regional":
        spaces["gamma_space"] = VectorFunctionSpace(mesh, "R", 0, dim = 17)
    else:
        gamma_family, gamma_degree = gamma_space.split("_")
        spaces["gamma_space"] = FunctionSpace(mesh, gamma_family, int(gamma_degree))
    spaces["displacement_space"] = VectorFunctionSpace(mesh, "CG", 2)
    spaces["pressure_space"] = FunctionSpace(mesh, "CG", 1)
    spaces["state_space"] = spaces["displacement_space"]*spaces["pressure_space"]
    spaces["strain_space"] = VectorFunctionSpace(mesh, "R", 0, dim=3)
    spaces["strainfield_space"] = VectorFunctionSpace(mesh, "CG", 1)
    spaces["quad_space"] = FunctionSpace(mesh, "Quadrature", 4)
    
    return spaces
def setup_moving_mesh(state_space, newmesh):
    V = state_space.sub(0).collapse()
    u_prev = Function(V)
    u_current = Function(V)
    state = Function(state_space)

    fa = FunctionAssigner(V, state_space.sub(0))
    
    d = Function(VectorFunctionSpace(newmesh,  "CG", 2))
    return u_prev, u_current, state, d, fa


def setup_bullseye_sim(bullseye_mesh, fun_arr):
    V = FunctionSpace(bullseye_mesh, "DG", 0)
    dm = V.dofmap()
    sfun = MeshFunction("size_t", bullseye_mesh, 2, bullseye_mesh.domains())

    funcs = []
    for time in range(len(fun_arr)):

        fun_tmp = Function(V)
        arr = fun_arr[time]

        for region in range(17):

                vertices = []

                for cell in cells(bullseye_mesh):

                    if sfun.array()[cell.index()] == region+1:

                        verts = dm.cell_dofs(cell.index())

                        for v in verts:
                            # Find the correct vertex index 
                            if v not in vertices:
                                vertices.append(v)

                fun_tmp.vector()[vertices] = arr[region]
        funcs.append(Vector(fun_tmp.vector()))
    return funcs
    




####### Compute stuff ###########
def recompute_strains_to_original_reference(strains, ref):

    strain_dict = {strain : {i:[] for i in STRAIN_REGION_NUMS}  for strain in STRAIN_NUM_TO_KEY.values()}
    
    for d in ['longitudinal', 'circumferential', 'radial']:
        for r in range(1,18):
            strain_trace = strains[d][r]
            new_strain_trace = np.zeros(len(strain_trace))
            ea0 = strain_trace[ref]
        
            for i in range(len(strain_trace)):
                                
                ei0 = strain_trace[i]
                eia = (ei0 - ea0)/(ea0 + 1)
                new_strain_trace[i] = eia

            new_strain_trace = np.roll(new_strain_trace, -ref)
            strain_dict[d][r] = new_strain_trace
            
    return strain_dict


def compute_inner_cavity_volume(mesh, ffun, endo_lv_marker):
    """
    Compute cavity volume using the divergence theorem

    *Arguments*
      mesh (:py:class:`dolfin.Mesh`)
        The mesh

      ffun (:py:class:`dolfin.MeshFunction`)
        Facet function

    *Return*
      vol (float)
        Volume of inner cavity 
    """
    
    X = SpatialCoordinate(mesh)
    N = FacetNormal(mesh)
    ds = Measure("exterior_facet", subdomain_data = ffun, domain = mesh)(endo_lv_marker)
    
    vol = assemble((-1.0/3.0)*dot(X, N)*ds)
    return vol




def get_mechanical_features(solver, state, gamma, spaces, data):

    solver.get_state().assign(state)
    solver.get_gamma().assign(gamma)

    # Fiber stress
    fiber_stress = project(solver.fiber_stress(), 
                           spaces["stress_space"])
    data["fiber_stresses"].append(Vector(fiber_stress.vector()))

    # Fiber strain
    fiber_strain = project(solver.fiber_strain(), 
                           spaces["stress_space"])
    data["fiber_strains"].append(Vector(fiber_strain.vector()))

    # Fiber Work
    fiber_work = project(solver.work_fiber(), 
                         spaces["stress_space"])
    data["fiber_work"].append(Vector(fiber_work.vector()))

    # Work
    work = project(solver.work(), 
                   spaces["stress_space"])
    data["work"].append(Vector(work.vector()))

    #Invariants
    I1 = project(solver.I1(), 
                 spaces["stress_space"])
    data["I1"].append(Vector(I1.vector()))

    I4f = project(solver.I4f(), 
                  spaces["stress_space"])
    data["I4f"].append(Vector(I4f.vector()))

    # Correct fiber strain and fiber stress
    Eff = solver.Eff(spaces["quad_space"])
    data["Eff"].append(Vector(Eff.vector()))

    Tff = solver.Tff(spaces["quad_space"])
    data["Tff"].append(Vector(Tff.vector()))


    
    

def get_regional(dx, fun, fun_lst, nobase = False, ub = np.inf, lb = -np.inf):

    
    if fun.value_size() == 17:
        return fun_lst.T

    meshvols = []
    
    if nobase:
        regions = range(6,17)
        i_start = 6
    else:
        regions = range(17)
        i_start = 0
    

    for i in regions:
        meshvols.append(Constant(assemble(Constant(1.0)*dx(i+1))))

    regional_lst = [[] for i in regions]
    for f in fun_lst:
        fun.vector()[:] = f
        for i in regions:
            regional_lst[i-i_start].append(min(max(lb, assemble((fun/meshvols[i-i_start])*dx(i+1))), ub))


 
    return regional_lst

def get_global(dx, fun, fun_lst, nobase = False):

    meshvols = []
    
    if nobase:
        regions = range(6,17)
        i_start = 6
    else:
        regions = range(17)
        i_start = 0
    

    for i in regions:
        meshvols.append(assemble(Constant(1.0)*dx(i+1)))

    meshvol = np.sum(meshvols)

    fun_mean = []
    for f in fun_lst:
        fun.vector()[:] = f
        # fun_mean.append(df.assemble((fun/meshvol)*dx))
        fun_tot = 0
        for i in regions:
            fun_tot += assemble((fun)*dx(i+1))

        fun_mean.append(fun_tot/meshvol)

 
    return fun_mean


def get_lst_norm(funcs, space, norm_type = "L2", mesh = None):
    fun = Function(space)

    norm_lst = []

    for f in funcs:
        fun.vector()[:] = f
        if norm_type == "linf":
            norm_lst.append(norm(f, norm_type, mesh))
        else:
            norm_lst.append(norm(fun, norm_type, mesh))

    return norm_lst

def get_norm_diff(funcs1, funcs2, space, norm_type = "L2"):

    assert len(funcs1) == len(funcs2)
        
    fun1 = Function(space)
    fun2 = Function(space)
        
    diff_arr = []
    norm_lst = []
    for f1, f2 in zip(funcs1, funcs2):
        fun1.vector()[:] = f1
        fun2.vector()[:] = f2

        diff_fun = project(fun1-fun2, space)
        norm_lst.append(norm(diff_fun, norm_type))

    return norm_lst

def get_regional_norm_lst(us, uhs, space, dx):

    u = Function(space)
    u_reg = get_regional(dx, u, us)
    uh_reg = get_regional(dx, u, uhs)

    u_diff  = np.abs(np.subtract(u_reg, uh_reg))

    max_u_reg = np.array([mi*np.ones(u_diff.shape[1]) for mi in np.max(u_reg, 1)])

    return np.sum(np.divide(u_diff, 17*max_u_reg), 0)

def get_maxnorm_lst(funcs1, funcs2, space):

    assert len(funcs1) == len(funcs2)
        
    fun1 = Function(space)
    fun2 = Function(space)
        
    diff_arr = []
    norm_lst = []
    for f1, f2 in zip(funcs1, funcs2):
        fun1.vector()[:] = f1
        fun2.vector()[:] = f2

        diff_fun = project(fun1-fun2, space)
        norm_lst.append(norm(diff_fun.vector(), "linf"))

    return norm_lst

def get_errornorm_lst(us, uhs, space, mesh, norm_type = "L2", degree_rise = 3):
    assert len(us) == len(uhs)
        
    u = Function(space)
    uh = Function(space)
        
    errornorm_lst = []
    for f1, f2 in zip(us, uhs):
        u.vector()[:] = f1
        uh.vector()[:] = f2
        
        errornorm_lst.append(errornorm(u,uh, norm_type, degree_rise, mesh))

    return errornorm_lst
def strain_to_arrs(strains1, strains2):

     # Put strain into arrays
    regions_sep = [range(1,7), range(7,13), range(13, 18)]
    dirs = ['circumferential','radial', 'longitudinal']
    

    strains = []
    t = 0
    for i in range(3):
        for region in regions_sep[i]:
            for direction in dirs:
                t+=1
                strains.append([strains1[direction][region],
                                strains2[direction][region]])
   
    return strains
                
def get_min_max_strain(strains_meas, strains_sim):

    s_maxs = []
    s_mins = []

    regions = range(1,18)
    for d in ['longitudinal', 'circumferential', 'radial']:

        strains_meas_dir = strains_meas[d]
        strains_sim_dir =  strains_sim[d]


        s_maxs.append(np.max(np.concatenate(([strains_meas_dir[r] for r in regions], 
                                             [strains_sim_dir[r] for r in regions]))))
      
        s_mins.append(np.min(np.concatenate(([strains_meas_dir[r] for r in regions], 
                                             [strains_sim_dir[r] for r in regions]))))

    return np.round(min(s_mins), 2), np.round(max(s_maxs), 2)

######## Plot stuff ################
def plot_regional(reg_arr, path, ylabel):


    basal = reg_arr[:6]
    mid = reg_arr[6:12]
    apical = reg_arr[12:]
    x = np.linspace(0,100, len(basal[0]))

    colors = ["b-", "r-", "g-", "c-", "m-", "y-"]
    titles = ["Basal", "Mid", "Apical"]

    fig, axes = plt.subplots(2,3)

    

    # for i, s in enumerate(basal[1:-1]):
    #     axes[0,0].plot(x, s, colors[i+1])
    #     axes[0,0].set_title(titles[0])

    # axes[0,0].set_ylabel(ylabel)

    # for k, stresses in enumerate([mid, apical]):

    #         for i, s in enumerate(stresses):
    #             axes[0,k+1].plot(x, s, colors[i])
    #             axes[0,k+1].set_title(titles[k+1])

    #         axes[0,0].set_ylabel(ylabel)

    
    for k, stresses in enumerate([basal, mid, apical]):

            for i, s in enumerate(stresses):
                axes[0,k].plot(x, s, colors[i])
                axes[0,k].set_title(titles[k])

            axes[0,0].set_ylabel(ylabel)

    basal_im = plt.imread("bullseye/bullseye_base_act.png")
    mid_im = plt.imread("bullseye/bullseye_mid_act.png")
    apical_im = plt.imread("bullseye/bullseye_apical_act.png")

    axes[1,0].set_axis_off()
    axes[1,0].imshow(basal_im)

    axes[1,1].set_axis_off()
    axes[1,1].imshow(mid_im)

    axes[1,2].set_axis_off()
    axes[1,2].imshow(apical_im)
    sns.despine()

    fig.savefig(path)      

def plot_canvas(strains, s_min, s_max, path, labels):

    # Set some seaborn styles
    sns.set_style("ticks")
    sns.set_palette(sns.husl_palette(2))
    sns.set_context("paper")
   
    # Create big plots so that we can gather 
    # the mid, basal and apical plots together
    fig, big_axes = plt.subplots(figsize=(15.0, 15.0) , nrows=3, ncols=1, 
                                 sharey=True, sharex = True)
 
    # Labels for the big 
    big_labels = ["Basal", "Mid", "Apical"]
    for row, big_ax in enumerate(big_axes, start=1):

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        # removes the white frame
        big_ax.set_frame_on(False)
        
        # Set the labels 
        big_ax.set_ylabel(big_labels[row-1], fontsize = 26)
        big_ax.yaxis.set_label_position("right")
        big_ax.yaxis.labelpad = 20

    # Titles for the minor plots
    regions = {1:"Anterior", 19:"Anterior", 37:"Anterior",
               2:"Anteroseptal", 20:"Anteroseptal",
               3:"Septum", 21:"Septum", 38:"Septum",
               4:"Inferior", 22:"Inferior", 39:"Inferior",
               5:"Posterior", 23:"Posterior",
               6:"Lateral", 24:"Lateral", 40:"Lateral", 41:"Apex"}

    # % of cardiac cycle
    x = np.linspace(0,100, len(strains[0][0]))

    # Some counter
    t = 0

    # Add subplots with strain plots
    for i in range(1,54):
        
        if i not in [42,48]:
            # Add a subplot
            ax = fig.add_subplot(9,6,i)
            
        # Put titles on the top ones at each level
        if i in range(1,7) + range(19,25) + range(37, 42):
            ax.set_title(r"{}".format(regions[i]) , fontsize = 16)
        
        # Put ticks on every one of them
        ax.set_ylim(s_min, s_max)
        ax.set_xlim(0,100)
        ax.set_yticks([s_min, 0, s_max])
        ax.set_xticks([0,50,100])

        # Put xlabels only on the bottom ones
        if i not in range(49,54) + [36]:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([0,50,100], fontsize = 12)
            ax.set_xlabel(r"$\%$ Cardiac Cycle", fontsize = 12)

        # Put y labels only on the left most ones
        if i not in range(1,50, 6):
            ax.set_ylabel("")
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([s_min, 0, s_max], fontsize = 12)
            if i in range(1, 50, 18):
                ax.set_ylabel(r"Circumferential", fontsize = 11)
            if i in range(7, 50, 18):
                ax.set_ylabel(r"Radial", fontsize = 11)
            if i in range(13, 50, 18):
                ax.set_ylabel(r"Longitudinal", fontsize = 11)


        # Plot the strain
        if i not in [42,48]:
            strain = strains[t]
            
            t+= 1
            # labels = ["Measured", "Simulated"]
            l1 = ax.plot(x, strain[0], label = labels[0])
            l2 = ax.plot(x, strain[1], label = labels[1])
            ax.axhline(y=0, ls = ":")
            lines = [l1[0], l2[0]]
    
        # Plot the legend
        if i == 48:
            ax = fig.add_subplot(9,6,i)
            ax.set_axis_off()
            ax.legend(lines, labels, "center", prop={'size':20})

    # Adjust size
    fig.tight_layout(w_pad=0.0)
    figw = 17.0
    figh = 17.0
    plt.subplots_adjust(left=1/figw, right=1-1/figw, bottom=1/figh, top=1-1/figh)
    
    # Remove top and right axis
    sns.despine()
    
    fig.savefig(path, bbox_inches='tight')
    plt.close()

def plot_strains2(simulated_strains, measured_strains, outdir):

    
    sns.set_style("ticks")
    sns.set_context("paper")
   
    ## Put the strain dictionaries in arrays
    
    # Basal, Mid, Apical
    # regions_sep = [range(1,7), range(7,13), range(13, 18)]
    regions_sep = [[1,3,4,6,5,2], [7,9,10,12,11,8], range(13, 18)]
    dirs = ['circumferential','radial', 'longitudinal']
    big_labels = ['Circumferential','Radial', 'Longitudinal']
    small_labels = [r"Basal", r"Mid", r"Apical"]
    labels = [r"Measured", r"Simulated"]

    # Array: Basal_regions Circ
    
    # smax = -np.inf 
    # smin = np.inf

    
    for d in range(3):
        direction = dirs[d]
        strains = []
        smaxs = []
        smins = []
        for i in range(3):

            smax = -np.inf 
            smin = np.inf
            for region in regions_sep[i]:
                cur_strain = [measured_strains[direction][region],
                              simulated_strains[direction][region]]

                strains.append(cur_strain)

                min_strain = np.min(cur_strain)
                max_strain = np.max(cur_strain)

                smax = max_strain if max_strain > smax else smax
                smin = min_strain if min_strain < smin else smin

            smaxs.append(smax)
            smins.append(smin)

        fig, big_ax = plt.subplots(figsize=(15.0, 6.0))

        # Labels for the big figure

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        # big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        # removes the white frame
        big_ax.set_frame_on(False)

        # Set the labels 
        big_ax.set_ylabel(big_labels[d], fontsize = 32)
        big_ax.yaxis.set_label_position("right")
        big_ax.yaxis.labelpad = 20


        regions = {1:"Anterior",
                   2:"Septum", 
                   3:"Inferior", 
                   4:"Lateral",
                   5:"Posterior",
                   6:"Anteroseptal",
                   7:"Apex"}

        # % of cardiac cycle
        x = np.linspace(0,100, len(strains[0][0]))

    
        t = 0
        # Add subplots with strain plots
        for i in range(1,22):

            if 1:
                ax = fig.add_subplot(3,7,i)

                # Put titles on the top ones at each level
                # if i in range(1,7) + range(19,25) + range(37, 42):
                #     ax.set_title(r"{}".format(regions[i]) , fontsize = 22)
                if d == 0:
                    if i in regions.keys():
                        ax.set_title(r"{}".format(regions[i]) , fontsize = 28, y = 1.1)

                if i in [7, 14, 19, 20]:
                    ax.set_axis_off()
                    continue

                # Put ticks on every one of them
                ax.set_xlim(0,100)
                ax.set_xticks([0,50,100])
                # ax.set_ylim(smin, smax)
                # ax.set_yticks([smin, 0, smax])


                # from IPython import embed; embed()
                # exit()
                if i <= 7:
                    ax.set_ylim(smins[0], smaxs[0])
                    ax.set_yticks([smins[0], 0, smaxs[0]])
                elif i > 14:
                    ax.set_ylim(smins[2], smaxs[2])
                    ax.set_yticks([smins[2], 0, smaxs[2]])

                else:
                    ax.set_ylim(smins[1], smaxs[1])
                    ax.set_yticks([smins[1], 0, smaxs[1]])

                # Put xlabels only on the bottom ones
                # if i not in range(49,54) + [36]:
                if i in [12,13,15, 16, 17, 18, 21] and d == 2: 
                    ax.set_xticklabels([0,50,100], fontsize = 22)
                    # ax.set_xlabel(r"$\%$ Cardiac Cycle", fontsize = 20)
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                # Put y labels only on the left most ones
                if i not in [1,8,15]:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
                else:

                    ax.set_yticklabels([smin, 0, smax], fontsize = 22)
                    if i == 1:
                        ax.set_ylabel(small_labels[0], fontsize = 28) # Basal
                    elif i == 8:
                        ax.set_ylabel(small_labels[1], fontsize = 28) # Mid
                    else:
                        ax.set_ylabel(small_labels[2], fontsize = 28) # Apical

                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

                
                strain = strains[t]
                t+= 1
                
                l1 = ax.plot(x, strain[0], "b-",label = labels[0])
                l2 = ax.plot(x, strain[1], "r-", label = labels[1])
                ax.axhline(y=0, ls = ":")
                lines = [l1[0], l2[0]]



            # Plot the legend
            if i == 7:
                ax = fig.add_subplot(3,7,21)
                ax.set_axis_off()
                ax.legend(lines, labels, "center", prop={'size':20})

        # Adjust size
        fig.tight_layout(w_pad = 0.0)
     
        # Remove top and right axis
        sns.despine()

        fig.savefig("{}/simulated_strain_{}.pdf".format(outdir, direction),
                    bbox_inches='tight')
        plt.close()
    

def plot_curves(x, ys, labels, title, xlabel, ylabel, path, legend_on = True, 
                logscale = False, small = False, save_legend = False, ylim = None):
    fig = plt.figure()
    ax = fig.gca()

    assert len(ys) == len(labels)
    
    lines = []
    for i in range(len(ys)):
        if logscale:
            line = ax.semilogy(x,ys[i], label = labels[i])
        else:
            line = ax.plot(x,ys[i] , label = labels[i], marker='o')

        lines.append(line[0])
    
    if small:
        
        plt.setp(lines, linewidth=2.0)

        plt.plot(x, np.zeros(len(x)), "k-")

        xlab = [np.min(x), np.median(x), np.max(x)]
        ylab = [np.min(ys), 0.0, np.max(ys)] if ylim is None else [ylim[0], 0.0, ylim[1]]
        
        plt.xticks(xlab)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
       
        plt.yticks(ylab)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.tick_params(axis='both', which='minor', labelsize=40)
        ax.grid(True)
        legend_on = False

    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.title(title)

    if legend_on:
        if len(ys) > 3:
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            ax.legend(loc = 0, prop={'size':15})
            fig.savefig(path)
    else:
        fig.savefig(path, bbox_inches='tight')

    if save_legend:
        figlegend = plt.figure(figsize=(4,2))
        plt.axis("off")
        figlegend.legend(lines, tuple(labels), "center")
        figlegend.savefig(path + "legend.pdf")
        
    sns.despine()
    plt.close()




#### VTK stuff ########
def ply_to_polydata(fname):
    
    reader = vtk.vtkPLYReader()
    reader.SetFileName(fname)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def write_to_vtp(fname, polydata):

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname+".vtp")

    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

def vtk_add_field(grid, fun) :

    V = fun.function_space()
    family = V.ufl_element().family()
    degree = V.ufl_element().degree()

    if fun.value_rank() > 0 :
        idx = np.column_stack([ V.sub(i).dofmap().dofs()
                    for i in xrange(0, V.num_sub_spaces()) ])
        fval = fun.vector().array()[idx]
    else :

        if family in ['Discontinuous Lagrange'] :
            fval = fun.vector().array()

        elif family in ['Real']:
            fval = fun.vector().array()[0]*np.ones(int(grid.GetNumberOfPoints()))

        elif family in ["Quadrature"]:
            
            # Take the average over the quadrature points within
            # a given cell. Visualize as pointwise cell averages
            fval_cell = np.zeros((V.mesh().num_cells(),))
            for c in cells(V.mesh()) :
                idx = c.index()
                fval_cell[idx] = fun.vector()[V.dofmap().cell_dofs(idx)].mean()

            # Visualize at the vertices by reducing the value at the
            # nearby quadrature points to one value at the vertex

            # Compute connvectivity between dimension 1 and 3
            V.mesh().init(1,3)
            fval_vert = np.zeros((V.mesh().num_vertices(),))
            for v in vertices(V.mesh()) :
                idx = v.index()
                # Find the cells who have v as a vertex
                c = V.mesh().topology()(1,3)(idx)
                # Find the dofs for the quadrature points within these cells
                dofs = reduce(np.union1d, [V.dofmap().cell_dofs(ci) for ci in c])
                # Take the average of the value at these dofs
                fval_vert[idx] = fun.vector()[dofs].mean()

            
            
       
        else:
            vtd = vertex_to_dof_map(V)
            fval_tmp = fun.vector().array()
            fval = np.zeros(len(fval_tmp))
            fval = fval_tmp[vtd]
           

    if fun.name() == 'displacement' :
        # add zero columns if necessary
        gdim = V.num_sub_spaces()
        fval = np.hstack([fval, np.zeros((fval.shape[0], 3-gdim))])


    if family in ['Discontinuous Lagrange']:
        funvtk = array2vtk(fval)
        funvtk.SetName(fun.name())
        grid.GetCellData().AddArray(funvtk)
        
    elif family in ["Quadrature"]:
        funvtk_cell = array2vtk(fval_cell)
        funvtk_cell.SetName(fun.name())
        grid.GetCellData().AddArray(funvtk_cell)

        funvtk_vert = array2vtk(fval_vert)
        funvtk_vert.SetName(fun.name())
        grid.GetPointData().AddArray(funvtk_vert)
        
        
    else :
        funvtk = array2vtk(fval)
        funvtk.SetName(fun.name())
        grid.GetPointData().AddArray(funvtk)



def dolfin2vtk(mesh):
    domain = mesh.ufl_domain()
    gdim = domain.geometric_dimension()
    mdim = domain.topological_dimension()
    order = 1
    # coordinates of the mesh
    coords = mesh.coordinates().copy()
    # connectivity
    conn = mesh.cells()

    # coords = np.hstack([coords, np.zeros((coords.shape[0], 3-gdim))])

    # only these are supported by dolfin
    vtk_shape = { 1 : { 1 : vtk.VTK_LINE,
                        2 : vtk.VTK_TRIANGLE,
                        3 : vtk.VTK_TETRA },
                  2 : { 1 : vtk.VTK_QUADRATIC_EDGE,
                        2 : vtk.VTK_QUADRATIC_TRIANGLE,
                        3 : vtk.VTK_QUADRATIC_TETRA } }[order][mdim]

    # create the grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(array2vtkPoints(coords))
    grid.SetCells(vtk_shape, array2vtkCellArray(conn))
    return grid




def write_pvd(pvd_name, fname, time_stamps):
    

    time_form = """<DataSet timestep="{}" part="0" file="{}" />"""

    body="""<?xml version="1.0"?>
    <VTKFile type="Collection" version="0.1">
    <Collection>
    {}
    </Collection>
    </VTKFile>
    """.format(" ".join(time_form.format(time_stamps[i],fname.format(i)) for i in range(len(time_stamps))))

    with open(pvd_name, "w") as f:
        f.write(body)

def write_to_vtk(grid, name):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInput(grid)
    writer.SetFileName(name)
    writer.Write()

def add_stuff(mesh, name, *args):
    grid = dolfin2vtk(mesh)

    for f in args:
        vtk_add_field(grid, f)

    write_to_vtk(grid, name)




if __name__ == "__main__":
    x = np.linspace(0,1,10)
    y1 = x
    y2 = np.multiply(x,x)

    ys = [y1, y2]
    plot_seaborn_curves(x,ys)

    
