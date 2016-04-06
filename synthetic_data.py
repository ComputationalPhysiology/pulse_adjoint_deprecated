from dolfin import *
from dolfin_adjoint import *

import numpy as np

from run_optimization import run_active_optimization, run_passive_optimization
from setup_optimization import make_solver_params, setup_adjoint_contraction_parameters, setup_general_parameters, initialize_patient_data, save_patient_data_to_simfile
from heart_problem import SyntheticHeartProblem
from adjoint_contraction_args import *
from store_opt_results import write_opt_results_to_h5

from numpy_mpi import *
from utils import contract_point_exists, passive_inflation_exists, Object, get_spaces, Text, pformat

CASES = ["matparams", "scalar_gamma", "continuous_gamma"]
def store_passive_inflation(phm, states, strains, strainfields, params):

    for_result_opt, ini_for_res = make_synthetic_results(phm)
    for_result_opt.states = states
    for_result_opt.strains = strains
    for_result_opt.strainfields = strainfields
    for_result_opt.strain_weights = phm.solver.parameters["strain_weights"]
    for_result_opt.strain_weights_deintegrated = phm.solver.parameters["strain_weights_deintegrated"]
    h5group =  PASSIVE_INFLATION_GROUP.format(params["alpha_matparams"])
    mesh = phm.solver.parameters["mesh"]
    opt_matparams = interpolate(Constant(params["Material_parameters"].values()), VectorFunctionSpace(mesh, "R", 0, dim=4))
    write_opt_results_to_h5(h5group, params, ini_for_res, for_result_opt, opt_matparams = opt_matparams)


def make_synthetic_results(phm):
    # Just make a lot of dummy variables for the writing to work
    for_result_opt = Object()
    for_result_opt.phm = phm
    for_result_opt.func_value_strain = 0.0
    for_result_opt.func_value_volume = 0.0
    for_result_opt.weighted_func_value_strain = 0.0
    for_result_opt.weighted_func_value_volume = 0.0
    for_result_opt.gamma_gradient = 0.0
    for_result_opt.reg_par = 0.0
    for_result_opt.strain_weights = np.ones((17, 3))
    for_result_opt.volumes = [0.0]
    for_result_opt.lv_pressures = [0.0]
    for_result_opt.rv_pressures = [0.0]
   

    ini_for_res =  Object()
    ini_for_res.func_value_strain = 0.0
    ini_for_res.func_value_volume = 0.0
    ini_for_res.weighted_func_value_strain = 0.0
    ini_for_res.weighted_func_value_volume = 0.0

    return for_result_opt, ini_for_res

def correct_for_drift(phm, synth_output, noise_arr):

    strains_original_lst, strains_noise_lst = add_noise_to_original_strains(phm, synth_output, noise_arr)

    mpi_print("Correct strains for drift")
    strains_corrected_lst = []
    for i in range(17):
        region_strain_noise = strains_noise_lst[i]
        c_noise = [s[0] for s in region_strain_noise]
        r_noise = [s[1] for s in region_strain_noise]
        l_noise = [s[2] for s in region_strain_noise]

        c_corr = subtract_line(c_noise)
        r_corr = subtract_line(r_noise)
        l_corr = subtract_line(l_noise)

        regions_strain_corr_all = np.asarray(zip(c_corr, r_corr, l_corr))
        strains_corrected_lst.append(regions_strain_corr_all)

    return strains_corrected_lst, strains_noise_lst, strains_original_lst
def add_noise_to_original_strains(phm, synth_output, noise_arr):
    
    strain_fun = Function(phm.strainspace)

    mpi_print("Add noise to original strains")
    strains_noise_lst  = [[] for i in range(17)]
    strains_original_lst  =[[] for i in range(17)]
    with HDF5File(mpi_comm_world(), synth_output, "r") as h5file:
        for point in range(len(noise_arr)):
            for i in STRAIN_REGION_NUMS:
                h5file.read(strain_fun.vector(), "/point_{}/original_strain/region_{}".format(point, i), True)
                strain_arr_orig = gather_broadcast(strain_fun.vector().array())
                strains_original_lst[i-1].append(strain_arr_orig)

                strain_arr_noise = np.add(strain_arr_orig, noise_arr[point])
                strains_noise_lst[i-1].append(strain_arr_noise)

    return strains_original_lst, strains_noise_lst

def add_noise_to_original_volumes(phm, synth_output, noise_arr):
    
    vol_fun = Function(FunctionSpace(phm.mesh, "R", 0))

    mpi_print("Add noise to original strains")
    vols_noise_lst  = []
    vols_original_lst = []
    
    with HDF5File(mpi_comm_world(), synth_output, "r") as h5file:
        for point in range(len(noise_arr)):
            h5file.read(vol_fun.vector(), "/point_{}/original_volume".format(point), True)
            vol_arr_orig = gather_broadcast(vol_fun.vector().array())[0]
            vols_original_lst.append(vol_arr_orig)
           

            vol_arr_noise = vol_arr_orig + noise_arr[point]
            vols_noise_lst.append(vol_arr_noise)

    return vols_original_lst, vols_noise_lst

def subtract_line(Y):    
    X = range(len(Y))
    # Create a linear interpolant between the first and the new point
    line = [ i*(Y[-1] - Y[0])/(len(X)-1) for i in X]

    # Subtract the line from the original points
    Y_sub = np.subtract(Y, line)
    return Y_sub

def store_synth_data(phm, storage, point, noise):

    scalar_func = Function(FunctionSpace(phm.solver.parameters["mesh"], "R", 0))
    h5group = "/point_{}".format(point)


    # Store strains
    for i in STRAIN_REGION_NUMS:
        if noise:
            storage.write(phm.strains[i-1].vector(), h5group + "/original_strain/region_{}".format(i))
        else:
            storage.write(phm.strains[i-1].vector(), h5group + "/strain/region_{}".format(i))
            

    storage.write(phm.strainfield, h5group + "/strainfield")

    # Store volume
    volume = phm.get_inner_cavity_volume()
    scalar_func.assign(Constant(volume))
    if noise:
        storage.write(scalar_func.vector(), h5group + "/original_volume")
    else:
        storage.write(scalar_func.vector(), h5group + "/volume")

    # Store pressure
    pressure = phm.p_lv.t
    scalar_func.assign(Constant(pressure))
    storage.write(scalar_func.vector(), h5group + "/pressure")

    # Store state
    storage.write(phm.solver.w, h5group + "/state")   

    # Store gamma
    storage.write(phm.gamma, h5group + "/activation_parameter") 

def make_synthetic_pressure(pressure):

    measurements = Object()
    pressure = np.multiply(KPA_TO_CPA, pressure)
    pressure = np.subtract(pressure, pressure[0])
    num_points = SYNTH_PASSIVE_FILLING + NSYNTH_POINTS + 1
    p1 = pressure[:(num_points/2+1)]
    p2 = np.linspace(p1[-1], 0, num_points/2)
    measurements.pressure = np.concatenate((p1,p2))

    return measurements


def generate_active_synth_data(params, gamma_expr, X0, patient):

    synth_output = params["outdir"] +  "/synth_data.h5"

    if mpi_comm_world().rank == 0:

        if not os.path.isdir(params["outdir"]):
            os.makedirs(params["outdir"])
        
        if os.path.exists(synth_output):
            os.remove(synth_output) 

    
    measurements = make_synthetic_pressure(patient.pressure)
    solver_parameters, p_lv = make_solver_params(params, patient, measurements)

    mesh = patient.mesh
    crl_basis = (patient.e_circ, patient.e_rad, patient.e_long)
    
    spaces = get_spaces(mesh)
    gamma_family, gamma_degree = params["gamma_space"].split("_")
    gamma_space = FunctionSpace(mesh, gamma_family, int(gamma_degree))

    gamma_zero =  Function(gamma_space)
    solver_parameters['material']['gamma'] = gamma_zero
    gamma_list = [gamma_zero]*(SYNTH_PASSIVE_FILLING-1)
    
    for x0 in X0:
        
        gamma = Expression(gamma_expr,x0 = x0)
        gamma_fun = project(gamma, gamma_space)
        gamma_list.append(gamma_fun)

    gamma = Expression('0.0')
    gamma_fun = project(gamma, gamma_space)
    gamma_list.append(gamma_fun)
   
    num_points = len(gamma_list)
    phm = SyntheticHeartProblem(measurements.pressure, solver_parameters, p_lv,  
                                patient.ENDO, crl_basis, spaces, gamma_list)
   
    # Storage for synth data
    storage = HDF5File(mpi_comm_world(),  synth_output, "w")
    # Store the first point
    store_synth_data(phm, storage, 0, params["noise"])
    
    it = 1
    # Run the forward model, and generate synthetic data
    logger.info("LV Pressure    LV Volume    Mean Strain Norms (c,r,l)")
    states = [Vector(phm.solver.w.vector())]
    strains = [[Vector(phm.strains[i].vector())] for i in range(17)]
    strainfields = []
    for sol, strain in phm:

        # Print output
        volume = phm.get_inner_cavity_volume()
        strain_list = [np.mean([gather_broadcast(phm.strains[s].vector().array())[0] 
                                for i in range(len(phm.strains))]) for s in range(3)]
        
        strain_str = ("{:.5f} "*len(strain_list)).format(*strain_list)
        logger.info(("{:.5f}" + " "*8 + "{:.5f}" + " "*5 + "{}").format(phm.p_lv.t, volume, strain_str))
        
        # Store the data
        store_synth_data(phm, storage, it, params["noise"])

        if it < SYNTH_PASSIVE_FILLING:
            # Store the states. To be used for passive inflation phase
            states.append(Vector(phm.solver.w.vector()))
            strainfields.append(Vector(phm.strainfield.vector()))
            for region in STRAIN_REGION_NUMS:
                # from IPython import embed; embed()
                strains[region-1].append(Vector(phm.strains[region-1].vector()))
            
        
        it += 1
    
    storage.close()
    # if not args.optimize_matparams:
    store_passive_inflation(phm, states, strains, strainfields, params)


    if params["noise"]:
        # Load the noise from generated file
        from yaml import load 
        with open(os.path.dirname(os.path.abspath(__file__))+"/noise_synth.yml", "rb" ) as f:
            noise = load(f)

        # Volume
        vols_noise = np.array(noise["vol_noise_list"])
        
        vols_original_lst, vols_noisy_lst = add_noise_to_original_volumes(phm, synth_output, vols_noise)
        
        scalar_func = Function(FunctionSpace(phm.solver.parameters["mesh"], "R", 0))
        mpi_print("Store the noisy volumes")
        with HDF5File(mpi_comm_world(), synth_output, "a") as h5file:
            for point in range(len(phm.pressure)):
                scalar_func.assign(Constant(vols_noisy_lst[point]))
                h5file.write(scalar_func.vector(), "/point_{}/volume_w_noise".format(point))
                h5file.write(scalar_func.vector(), "/point_{}/volume".format(point))

        #Strain
        strain_noise = np.array(noise["strain_noise_list"])
        
        # Correct noisy strain curves by subtracting a line so that first and last point have zero strain
        strains_corrected_lst, strains_noisy_lst, strains_original_lst = correct_for_drift(phm, synth_output, strain_noise) 
        s = Function(phm.strainspace)
        mpi_print("Store the corrected strains")
        with HDF5File(mpi_comm_world(), synth_output, "a") as h5file:
            for point in range(len(phm.pressure)):
                for i in STRAIN_REGION_NUMS:
                    # Corrected
                    assign_to_vector(s.vector(),strains_corrected_lst[i-1][point])
                    h5file.write(s.vector(), "/point_{}/corrected_strain/region_{}".format(point, i))
                    # Overwrite the originals. These are the ones we are trying to get
                    h5file.write(s.vector(), "/point_{}/strain/region_{}".format(point, i))

                    # Strains with noise
                    assign_to_vector(s.vector(),strains_noisy_lst[i-1][point])
                    h5file.write(s.vector(), "/point_{}/strain_w_noise/region_{}".format(point, i))



def run_active_synth_data(params):

    setup_general_parameters()
    

    params["phase"] = "all"
    params["synth_data"] = True
    patient = initialize_patient_data(params["Patient_parameters"], synth_data = True)
    save_patient_data_to_simfile(patient, params["sim_file"])
    logger.info(pformat(params.to_dict()))

    # Define gamma as a gaussian function
    gamma_expr = '0.1*exp(-((x[0]-x0)*(x[0]-x0))/10)*(1- exp( -10*x[0]*x[0]))'
    
    # Move the concentration of gaussian function from apex to base
    # in order to get an apex-to-base contraction pattern
    X0 = np.linspace(10,-2, NSYNTH_POINTS)

        
    # if not passive_inflation_exists(args_synth):
    mpi_print('\nGenerate synthetic data:')
    generate_active_synth_data(params, gamma_expr, X0, patient)
    mpi_print('\nDone generating synthetic data... run optimization')
  
    params["phase"] = PHASES[1]  

    run_active_optimization(params, patient)
    




def test_matparms_synth_data():
    
    # Parser for generating synthetic data 
    matparam_parser_synth = make_material_param_parser()
    parser_synth = argparse.ArgumentParser(parents = [matparam_parser_synth], description = \
    'Generate synthetic data for material parameter testing')
    
    args_synth, case = parser_synth.parse_known_args()
    # Use these material parameters
    par = [0.6, 5.0, 20.0, 38.0]
    args_synth.initial_params =  par
    print '\nGenerate synthetic data with parameters'
    print 'a = {}, b = {}, a_f = {}, b_f = {}\n'.format(par[0], par[1], par[2], par[3])
    
    # Generate sythetic data
    generate_syntethic_data_matparams(args_synth)

    
    # Parser for the optimization
    mat_parser = make_material_param_parser()
    myparser = argparse.ArgumentParser(parents = [mat_parser], description = \
                                       'Calibrate Passive Material Parameters using Patient Data from Impact Study')
    args, case= myparser.parse_known_args()
    args.synth_data = True

    # Initial parameters
    par_init =  [0.732, 7.362, 21.51, 40.02]
    args.initial_params = par_init

    print '\nRun optimization with initial parameters'
    print 'a = {}, b = {}, a_f = {}, b_f = {}\n'.format(par_init[0], par_init[1], par_init[2], par_init[3])
    
    adj_reset()
    run_optimization(args)





if __name__ == "__main__":

    
    params = setup_adjoint_contraction_parameters()
    params["noise"] = True
    run_active_synth_data(params)
