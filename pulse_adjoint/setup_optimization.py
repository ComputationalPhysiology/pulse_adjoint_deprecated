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
from dolfinimport import *
import numpy as np
from utils import Object, Text, print_line, print_head
from adjoint_contraction_args import *
from numpy_mpi import *


def check_parameters(params):
    """Check that parameters are consistent.
    If not change the parameters and print out
    a warning

    :param params: Application parameters

    """

    mesh_type = params["Patient_parameters"]["mesh_type"]
    
    if mesh_type == "lv":
    
        if params["Optimization_targets"]["rv_volume"]:
            logger.warning("Cannot optimize RV volume using an LV geometry")
            params["Optimization_targets"]["rv_volume"] = False
            

def setup_adjoint_contraction_parameters():

    params = setup_application_parameters()

    # Patient parameters
    patient_parameters = setup_patient_parameters()
    params.add(patient_parameters)

    # Optimization parameters
    opt_parameters = setup_optimization_parameters()
    params.add(opt_parameters)

    # Optimization targets
    opttarget_parameters = setup_optimizationtarget_parameters()
    params.add(opttarget_parameters)

    # Weigths for each optimization target
    optweigths_active_parameters = setup_active_optimization_weigths()
    params.add(optweigths_active_parameters)
    optweigths_passive_parameters = setup_passive_optimization_weigths()
    params.add(optweigths_passive_parameters)

    check_parameters(params)
    
    return params
    
    
def setup_solver_parameters():
    """
    Have a look at `dolfin.NonlinearVariationalSolver.default_parameters`
    for options

    """
    
    solver_parameters = {"snes_solver":{}}

    solver_parameters["nonlinear_solver"] = "snes"
    solver_parameters["snes_solver"]["method"] = "newtontr"
    solver_parameters["snes_solver"]["maximum_iterations"] = 15
    solver_parameters["snes_solver"]["absolute_tolerance"] = 1.0e-5
    solver_parameters["snes_solver"]["linear_solver"] = "lu"
    
    

    return solver_parameters
    
   

def setup_general_parameters():
    """
    Parameters to speed up the compiler
    """

    # Parameter for the compiler
    flags = ["-O3", "-ffast-math", "-march=native"]
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    # dolfin.parameters["adjoint"]["test_derivative"] = True
    # dolfin.parameters["std_out_all_processes"] = False
    # dolfin.parameters["num_threads"] = 8
    
    dolfin.set_log_active(False)
    dolfin.set_log_level(INFO)


def setup_patient_parameters():
    """
    Have a look at :py:class:`patient_data.FullPatient`
    for options

    Defaults are

    +------------------+-----------------+---------------+
    | key              | Default Value   | Description   |
    +==================+=================+===============+
    | weight_rule      | equal           |               |
    +------------------+-----------------+---------------+
    | patient          | Joakim          |               |
    +------------------+-----------------+---------------+
    | weight_direction | all             |               |
    +------------------+-----------------+---------------+
    | include_sheets   | False           |               |
    +------------------+-----------------+---------------+
    | patient_type     | full            |               |
    +------------------+-----------------+---------------+
    | mesh_path        |                 |               |
    +------------------+-----------------+---------------+
    | fiber_angle_epi  | -60             |               |
    +------------------+-----------------+---------------+
    | subsample        | False           |               |
    +------------------+-----------------+---------------+
    | mesh_type        | lv              |               |
    +------------------+-----------------+---------------+
    | pressure_path    |                 |               |
    +------------------+-----------------+---------------+
    | echo_path        |                 |               |
    +------------------+-----------------+---------------+
    | resolution       | low_res         |               |
    +------------------+-----------------+---------------+
    | fiber_angle_endo | 60              |               |
    +------------------+-----------------+---------------+

    """
    
    params = Parameters("Patient_parameters")
    params.add("patient", "Joakim")
    params.add("patient_type", "full")
    params.add("weight_rule", DEFAULT_WEIGHT_RULE, WEIGHT_RULES)
    params.add("weight_direction", DEFAULT_WEIGHT_DIRECTION, WEIGHT_DIRECTIONS) 
    params.add("resolution", "low_res")
    params.add("pressure_path", "")
    params.add("mesh_path", "")
    params.add("echo_path", "")
    
    params.add("subsample", False)
    params.add("fiber_angle_epi", -60)
    params.add("fiber_angle_endo", 60)
    params.add("mesh_type", "lv", ["lv", "biv"])
    params.add("include_sheets", False)

    return params

def setup_optimizationtarget_parameters():
    """
    Set which targets to use
    Default solver parameters are:

    +----------------------+-----------------------+
    |Key                   | Default value         |
    +======================+=======================+
    | volume               | True                  |
    +----------------------+-----------------------+
    | rv_volume            | False                 |
    +----------------------+-----------------------+
    | regional_strain      | True                  |
    +----------------------+-----------------------+
    | full_strain          | False                 |
    +----------------------+-----------------------+
    | GL_strain            | False                 |
    +----------------------+-----------------------+
    | GC_strain            | False                 |
    +----------------------+-----------------------+
    | displacement         | False                 |
    +----------------------+-----------------------+
    
    """

    params = Parameters("Optimization_targets")
    params.add("volume", True)
    params.add("rv_volume", False)
    params.add("regional_strain", True)
    params.add("full_strain", False)
    params.add("GL_strain", False)
    params.add("GC_strain", False)
    params.add("displacement", False)
    return params

def setup_active_optimization_weigths():
    """
    Set the weight on each target (if used) for the active phase.
    Default solver parameters are:

    +----------------------+-----------------------+
    |Key                   | Default value         |
    +======================+=======================+
    | volume               | 0.95                  |
    +----------------------+-----------------------+
    | rv_volume            | 0.95                  |
    +----------------------+-----------------------+
    | regional_strain      | 0.05                  |
    +----------------------+-----------------------+
    | full_strain          | 1.0                   |
    +----------------------+-----------------------+
    | GL_strain            | 0.05                  |
    +----------------------+-----------------------+
    | GC_strain            | 0.05                  |
    +----------------------+-----------------------+
    | displacement         | 1.0                   |
    +----------------------+-----------------------+
    | regularization       | 0.01                  |
    +----------------------+-----------------------+
    
    """
    params = Parameters("Active_optimization_weigths")
    
    
    params.add("volume", 0.95)
    params.add("rv_volume", 0.95)
    params.add("regional_strain", 0.05)
    params.add("full_strain", 1.0)
    params.add("GL_strain", 0.05)
    params.add("GC_strain", 0.05)
    params.add("displacement", 1.0)
    params.add("regularization", 0.01)
        
    return params

def setup_passive_optimization_weigths():
    """
    Set the weight on each target (if used) for the passive phase.
    Default solver parameters are:

    +----------------------+-----------------------+
    |Key                   | Default value         |
    +======================+=======================+
    | volume               | 1.0                   |
    +----------------------+-----------------------+
    | rv_volume            | 1.0                   |
    +----------------------+-----------------------+
    | regional_strain      | 0.0                   |
    +----------------------+-----------------------+
    | full_strain          | 1.0                   |
    +----------------------+-----------------------+
    | GL_strain            | 0.05                  |
    +----------------------+-----------------------+
    | GC_strain            | 0.05                  |
    +----------------------+-----------------------+
    | displacement         | 1.0                   |
    +----------------------+-----------------------+
    | regularization       | 0.0                   |
    +----------------------+-----------------------+
    
    """
    
    params = Parameters("Passive_optimization_weigths")
    
    params.add("volume", 1.0)
    params.add("rv_volume", 1.0)
    params.add("regional_strain", 0.0)
    params.add("full_strain", 1.0)
    params.add("GL_strain", 0.05)
    params.add("displacement", 1.0)
    params.add("regularization", 0.0)
    
    return params
    
def setup_application_parameters():
    """
    Setup the main parameters for the pipeline

    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | key                                 | Default Value                                        | Description                        |
    +=====================================+======================================================+====================================+
    | base_bc                             | 'fix_x'                                              | Boudary condition at the base.     |
    |                                     |                                                      | ['fix_x', 'fixed', 'from_seg_base] |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | matparams_space                     | 'R_0'                                                | Space for material parameters.     |
    |                                     |                                                      | 'R_0', 'regional' or 'CG_1'        |         
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | noise                               | False                                                | If synthetic data, add noise       |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | use_deintegrated_strains            | False                                                | Use full strain field              |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | nonzero_initial_guess               | True                                                 | If true, use gamma = 0 as initial  |
    |                                     |                                                      | guess for all iterations           |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | active_model                        | 'active_strain'                                      | 'active_strain', 'active stress'   |
    |                                     |                                                      | or 'active_strain_rossi'           |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | base_spring_k                       | 1.0                                                  | Basal spring constant              |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | synth_data                          | False                                                | Synthetic data                     |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | sim_file                            | 'result.h5'                                          | Path to result file                |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | Material_parameters                 | {'a': 2.28, 'a_f': 1.685, 'b': 9.726, 'b_f': 15.779} | Material parameters                |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | phase                               | passive_inflation                                    | 'passive_inflation'                |
    |                                     |                                                      | 'active_contraction' or 'all'      |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | optimize_matparams                  | True                                                 | Optimiza materal parameter or use  |
    |                                     |                                                      | default values                     |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | state_space                         | 'P_2:P_1'                                            | Taylor-hood finite elemet          |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | gamma_space                         | 'CG_1'                                               | Space for gammma.                  |
    |                                     |                                                      | 'R_0', 'regional' or 'CG_1'        |         
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | incomp_penalty                      | 0.0                                                  | Penalty for compresssible model    |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | compressibility                     | 'incompressible'                                     | Model for compressibility          |
    |                                     |                                                      | see compressibility.py             |         
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | active_contraction_iteration_number | 0                                                    | Iteration in the active phase      |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | outdir                              |                                                      | Direction for the result           |
    +-------------------------------------+------------------------------------------------------+------------------------------------+

    """
    params = Parameters("Application_parmeteres")

    ## Output ##
    
    # Location of output
    params.add("sim_file", "result.h5")
    params.add("outdir", os.path.dirname(params["sim_file"]))

    ## Parameters ##
    
    # Spring constant at base (Note: works one for base_bc = fix_x)
    params.add("base_spring_k", 1.0)

    # Material parameters
    material_parameters = Parameters("Material_parameters")
    material_parameters.add("a", 2.28)
    material_parameters.add("a_f", 1.685)
    material_parameters.add("b", 9.726)
    material_parameters.add("b_f", 15.779)
    params.add(material_parameters)
    
    # Space for material parameter(s)
    # If optimization of multiple material parameters are selected,
    # then R_0 is currently the only applicable space
    params.add("matparams_space", "R_0", ["CG_1", "R_0", "regional"])
    

    ## Models ##

    # Active model
    params.add("active_model", "active_stress", ["active_strain",
                                                 "active_strain_rossi",
                                                 "active_stress"])


    # State space
    params.add("state_space", "P_2:P_1")

    # Model for compressibiliy
    params.add("compressibility", "incompressible", ["incompressible", 
                                                     "stabalized_incompressible", 
                                                     "penalty", "hu_washizu"])
    # Incompressibility penalty (applicable if model is not incompressible)
    params.add("incompressibility_penalty", 0.0)

    # Boundary condition at base
    params.add("base_bc", "fix_x", ["from_seg_base",
                                    "fix_x",
                                    "fixed"])


    ## Iterators ##

    # Active of passive phase
    params.add("phase", PHASES[0], PHASES)

    # Iteration for active phase
    params.add("active_contraction_iteration_number", 0)
    

    ## Additional setup ##

    # For passive optimization, include all passive points ('all')
    # or only the final point ('final')
    params.add("passive_weights", "all", ["final", "all"])
    
    # Update weights so that the initial value of the functional is 0.1
    params.add("adaptive_weights", True)
    
    # Space for active parameter
    params.add("gamma_space", "CG_1", ["CG_1", "R_0", "regional"])
    
    # If you want to use pointswise strains as input (only synthetic)
    params.add("use_deintegrated_strains", False)

    # If you want to optimize passive parameters
    params.add("optimize_matparams", True)

    # If you want to use a zero initial guess for gamma (False),
    # or use gamma from previous iteration as initial guess (True)
    params.add("nonzero_initial_guess", True)

    # Use synthetic data
    params.add("synth_data", False)
    # Noise is added to synthetic data
    params.add("noise", False)

    params.add("log_level", logging.INFO)


    return params

def setup_optimization_parameters():
    """
    Parameters for the optimization.
    Default parameters are

    +-----------------+-----------------+---------------+
    | key             | Default Value   | Description   |
    +=================+=================+===============+
    | disp            | False           |               |
    +-----------------+-----------------+---------------+
    | active_maxiter  | 100             |               |
    +-----------------+-----------------+---------------+
    | scale           | 1.0             |               |
    +-----------------+-----------------+---------------+
    | passive_maxiter | 30              |               |
    +-----------------+-----------------+---------------+
    | matparams_max   | 50.0            |               |
    +-----------------+-----------------+---------------+
    | fix_a           | False           |               |
    +-----------------+-----------------+---------------+
    | fix_a_f         | True            |               |
    +-----------------+-----------------+---------------+
    | fix_b           | True            |               |
    +-----------------+-----------------+---------------+
    | fix_b_f         | True            |               |
    +-----------------+-----------------+---------------+
    | gamma_max       | 0.9             |               |
    +-----------------+-----------------+---------------+
    | matparams_min   | 0.1             |               |
    +-----------------+-----------------+---------------+
    | passive_opt_tol | 1e-06           |               |
    +-----------------+-----------------+---------------+
    | active_opt_tol  | 1e-06           |               |
    +-----------------+-----------------+---------------+
    | method_1d       | brent           |               |
    +-----------------+-----------------+---------------+
    | method          | slsqp           |               |
    +-----------------+-----------------+---------------+
    

    """
    # Parameters for the Optimization
    params = Parameters("Optimization_parmeteres")
    params.add("method", "slsqp")
    params.add("method_1d", "brent")
    params.add("active_opt_tol", 1e-6)
    params.add("active_maxiter", 100)
    params.add("passive_opt_tol", 1e-6)
    params.add("passive_maxiter", 30)
    params.add("scale", 1.0)
    params.add("gamma_max", 0.9)
    params.add("matparams_min", 0.1)
    params.add("matparams_max", 50.0)
    params.add("fix_a", False)
    params.add("fix_a_f", True)
    params.add("fix_b", True)
    params.add("fix_b_f", True)
   
    
    params.add("disp", False)

    return params


def initialize_patient_data(patient_parameters, synth_data=False):
    """
    Make an instance of patient from :py:module`patient_data`
    baed on th given parameters

    **Example of usage**::
    
      params = setup_patient_parameters()
      patient = initialize_patient_data(params, False)

    :param dict patient_parameters: the parameters 
    :param bool synth_data: If synthetic data or not
    :returns: A patient instance
    :rtype: :py:class`patient_data.Patient`

    """
    
    logger.info("Initialize patient data")
    from patient_data import Patient
    
    patient = Patient(**patient_parameters)

    if synth_data:
        patient.passive_filling_duration = SYNTH_PASSIVE_FILLING
        patient.num_contract_points =  NSYNTH_POINTS + 1
        patient.num_points = SYNTH_PASSIVE_FILLING + NSYNTH_POINTS + 1

    return patient

def save_patient_data_to_simfile(patient, sim_file):

    file_format = "a" if os.path.isfile(sim_file) else "w"
    from mesh_generation.mesh_utils import save_geometry_to_h5

    fields = []
    for att in ["e_f", "e_s", "e_sn"]:
        if hasattr(patient, att):
            fields.append(getattr(patient, att))

    local_basis = []
    for att in ["e_circ", "e_rad", "e_long"]:
        if hasattr(patient, att):
            local_basis.append(getattr(patient, att))
    
    save_geometry_to_h5(patient.mesh, sim_file, "", patient.markers,
                            fields, local_basis)


def load_synth_data(mesh, synth_output, num_points, use_deintegrated_strains = False):
    pressure = []
    volume = []
    
    strainfieldspace = VectorFunctionSpace(mesh, "CG", 1, dim = 3)
    strain_deintegrated = []

    strain_fun = Function(VectorFunctionSpace(mesh, "R", 0, dim = 3))
    scalar_fun = Function(FunctionSpace(mesh, "R", 0))

    c = [[] for i in range(17)]
    r = [[] for i in range(17)]
    l = [[] for i in range(17)]
    with HDF5File(mpi_comm_world(), synth_output, "r") as h5file:
       
        for point in range(num_points):
            assert h5file.has_dataset("point_{}".format(point)), "point {} does not exist".format(point)
            # assert h5file.has_dataset("point_{}/{}".format(point, strain_group)), "invalid strain group, {}".format(strain_group)

            # Pressure
            h5file.read(scalar_fun.vector(), "/point_{}/pressure".format(point), True)
            p = gather_broadcast(scalar_fun.vector().array())
            pressure.append(p[0])

            # Volume
            h5file.read(scalar_fun.vector(), "/point_{}/volume".format(point), True)
            v = gather_broadcast(scalar_fun.vector().array())
            volume.append(v[0])

            # Strain
            for i in STRAIN_REGION_NUMS:
                h5file.read(strain_fun.vector(), "/point_{}/strain/region_{}".format(point, i), True)
                strain_arr = gather_broadcast(strain_fun.vector().array())
                c[i-1].append(strain_arr[0])
                r[i-1].append(strain_arr[1])
                l[i-1].append(strain_arr[2])

            if use_deintegrated_strains:
                strain_fun_deintegrated = Function(strainfieldspace, name = "strainfield_point_{}".format(point))
                h5file.read(strain_fun_deintegrated.vector(), "/point_{}/strainfield".format(point), True)
                strain_deintegrated.append(Vector(strain_fun_deintegrated.vector()))

    # Put the strains in the right format
    strain = {i:[] for i in STRAIN_REGION_NUMS }
    for i in STRAIN_REGION_NUMS:
        strain[i] = zip(c[i-1], r[i-1], l[i-1])

    if use_deintegrated_strains:
        strain = (strain, strain_deintegrated)
        
    return pressure, volume, strain


def get_simulated_strain_traces(phm):
        simulated_strains = {strain : np.zeros(17) for strain in STRAIN_NUM_TO_KEY.values()}
        strains = phm.strains
        for direction in range(3):
            for region in range(17):
                simulated_strains[STRAIN_NUM_TO_KEY[direction]][region] = gather_broadcast(strains[region].vector().array())[direction]
        return simulated_strains

def make_solver_params(params, patient, measurements):

    
    ##  Contraction parameter
    if params["gamma_space"] == "regional":
        gamma = RegionalParameter(patient.strain_markers)
    else:
        gamma_family, gamma_degree = params["gamma_space"].split("_")
        gamma_space = FunctionSpace(patient.mesh, gamma_family, int(gamma_degree))

        gamma = Function(gamma_space, name = 'activation parameter')

        

    ##  Material parameters

    # Number of passive parameters to optimize
    fixed_matparams_keys = ["fix_a", "fix_a_f", "fix_b", "fix_b_f"]
    npassive = sum([ not params["Optimization_parmeteres"][k] \
                     for k in fixed_matparams_keys])

    
    # Create an object for each single material parameter
    if params["matparams_space"] == "regional":
        paramvec_ = RegionalParameter(patient.strain_markers)
        
    else:
        
        family, degree = params["matparams_space"].split("_")
        matparams_space = FunctionSpace(patient.mesh, family, int(degree))
        paramvec_ = Function(matparams_space, name = "matparam vector")

        
    if npassive <= 1:
        # If there is only one parameter, just pick the same object
        paramvec = paramvec_

        # If there is none then 
        if npassive == 0:
            logger.debug("All material paramters are fixed")
            params["optimize_matparams"] = False

    else:
        
        # Otherwise, we make a mixed parameter
        paramvec = MixedParameter(paramvec_, npassive)
        # Make an iterator for the function assigment
        nopts_par = 0


    if params["phase"] in [PHASES[1]]:
        # Load the parameters from the result file  
                
        # Open simulation file
        with HDF5File(mpi_comm_world(), params["sim_file"], 'r') as h5file:
            
            # Get material parameter from passive phase file
            h5file.read(paramvec, PASSIVE_INFLATION_GROUP + "/optimal_control")
            
            
    matparams = params["Material_parameters"].to_dict()
    for par, val in matparams.iteritems():

        # Check if material parameter should be fixed
        if not params["Optimization_parmeteres"]["fix_{}".format(par)]:
            # If not, then we need to put the parameter into some dolfin function

            
            # Use the materal parameters from the parameters as initial guess
            if params["phase"] in [PHASES[0], PHASES[2]]:

                
                val_const = Constant(val) if paramvec_.value_size() == 1 \
                            else Constant([val]*paramvec_.value_size())
                

                if npassive <= 1:
                    paramvec.assign(val_const)

                else:
                    paramvec.assign_sub(val_const, nopts_par)
                
                    
            if npassive <= 1:
                matparams[par] = paramvec

            else:
                matparams[par] = split(paramvec)[nopts_par]
                nopts_par += 1

            
                    
                
            

   
    # Print the material parameter to stdout
    logger.info("\nMaterial Parameters")
    nopts_par = 0

    for par, v in matparams.iteritems():
        if isinstance(v, (float, int)):
            logger.info("\t{}\t= {:.3f}".format(par, v))
        else:
            
            if npassive <= 1:
                v_ = gather_broadcast(v.vector().array())
                
            else:
                v_ = gather_broadcast(paramvec.split(deepcopy=True)[nopts_par].vector().array())
                nopts_par += 1
            
            sp_str = "(mean), spatially resolved" if len(v_) > 1 else ""
            logger.info("\t{}\t= {:.3f} {}".format(par, v_.mean(), sp_str))

    
    ##  Material model
    from material import HolzapfelOgden
    
    if params["active_model"] == "active_strain_rossi":
        material = HolzapfelOgden(patient.e_f, gamma, matparams, params["active_model"],
                                      patient.strain_markers, s0 = patient.e_s, n0 = patient.e_sn)
    else:
        material = HolzapfelOgden(patient.e_f, gamma, matparams,
                                  params["active_model"], patient.strain_markers)

    


    strain_weights = None if not hasattr(patient, "strain_weights") else patient.strain_weights

    
    strain_weights_deintegrated = patient.strain_weights_deintegrated \
      if params["use_deintegrated_strains"] else None
    
        

    # Neumann BC
    neuman_bc = []

    V_real = FunctionSpace(patient.mesh, "R", 0)
    p_lv = Expression("t", t = measurements["pressure"][0],
                      name = "LV_endo_pressure", element = V_real.ufl_element())
    N = FacetNormal(patient.mesh)

    if patient.mesh_type() == "biv":
        p_rv = Expression("t", t = measurements["rv_pressure"][0],
                          name = "RV_endo_pressure", element = V_real.ufl_element())
        
        neumann_bc = [[p_lv, patient.ENDO_LV],
                     [p_rv, patient.ENDO_RV]]

        pressure = {"p_lv":p_lv, "p_rv":p_rv}
    else:
        neumann_bc = [[p_lv, patient.ENDO]]
        pressure = {"p_lv":p_lv}
    

    # Direchlet BC at the Base
    try:
        mesh_verts = patient.mesh_verts
        seg_verts = measurements.seg_verts
    except:
        logger.debug("No mesh vertices found. Fix base is the only applicable Direchlet BC")
        mesh_verts = None


    if params["base_bc"] == "from_seg_base" and (mesh_verts is not None):

        endoring = VertexDomain(mesh_verts)
        base_it = Expression("t", t = 0.0, name = "base_iterator")
        
        robin_bc = [None]
       
        # Expression for defining the boundary conditions
        base_bc_y = BaseExpression(mesh_verts, seg_verts, "y", base_it, name = "base_expr_y")
        base_bc_z = BaseExpression(mesh_verts, seg_verts, "z", base_it, name = "base_expr_z")
            
        def base_bc(W):
            """
            Fix base in the x = 0 plane, and fix the vertices at 
            the endoring at the base according to the segmeted surfaces. 
            """
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)

        
            bc = [DirichletBC(V.sub(0), Constant(0.0), patient.BASE),
                      DirichletBC(V.sub(1), base_bc_y, endoring, "pointwise"),
                      DirichletBC(V.sub(2), base_bc_z, endoring, "pointwise")]
            return bc

    elif params["base_bc"] == "fixed":
        
        robin_bc = [None]
        base_bc_y = None
        base_bc_z = None
        base_it = None
        
        def base_bc(W):
            '''Fix the basal plane.
            '''
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            bc = [DirichletBC(V, Constant((0, 0, 0)), patient.BASE)]
            return bc
        
        
    else:
        if not params["base_bc"] == "fix_x":
            if mesh_verts is None:
                logger.warning("No mesh vertices found. This must be set in the patient class")
            else:
                logger.warning("Unknown Base BC")
            logger.warning("Fix base in x direction")
    
        def base_bc(W):
            '''Make Dirichlet boundary conditions where the base is allowed to slide
            in the x = 0 plane.
            '''
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            bc = [DirichletBC(V.sub(0), 0, patient.BASE)]
            return bc
    
        
        base_bc_y = None
        base_bc_z = None
        base_it = None
        
        # Apply a linear sprint robin type BC to limit motion
        robin_bc = [[-Constant(params["base_spring_k"], 
                                   name ="base_spring_constant"), patient.BASE]]



    # Circumferential, Radial and Longitudinal basis vector
    crl_basis = {}
    for att in ["e_circ", "e_rad", "e_long"]:
        if hasattr(patient, att):
            crl_basis[att] = getattr(patient, att)

    
    
    solver_parameters = {"mesh": patient.mesh,
                         "facet_function": patient.facets_markers,
                         "facet_normal": N,
                         "crl_basis":crl_basis,
                         "passive_filling_duration": patient.passive_filling_duration, 
                         "mesh_function": patient.strain_markers,
                         "base_bc_y":base_bc_y,
                         "base_bc_z":base_bc_z,
                         "base_it":base_it,
                         "markers":patient.markers,
                         "strain_weights": strain_weights, 
                         "strain_weights_deintegrated": strain_weights_deintegrated,
                         "state_space": "P_2:P_1",
                         "compressibility":{"type": params["compressibility"],
                                            "lambda": params["incompressibility_penalty"]},
                         "material": material,
                         "bc":{"dirichlet": base_bc,
                               "neumann":neumann_bc,
                               "robin": robin_bc},
                         "solve":setup_solver_parameters()}


    if params["phase"] in [PHASES[0], PHASES[2]]:
        return solver_parameters, pressure, paramvec
    elif params["phase"] == PHASES[1]:
        return solver_parameters, pressure, gamma
    else:
        return solver_parameters, pressure

def get_measurements(params, patient):
    """Get the measurement or the synthetic data
    to be used as BC or targets in the optimization

    :param params: Application parameter
    :param patient: class with the patient data
    :returns: The target data
    :rtype: dict

    """
    
    # Find the start and end of the measurements
    if params["phase"] == PHASES[0]: #Passive inflation
        # We need just the points from the passive phase
        start = 0
        end = patient.passive_filling_duration

    elif params["phase"] == PHASES[1]: #Scalar contraction
        # We need just the points from the active phase
        start = patient.passive_filling_duration -1
        end = patient.num_points
      
    else:
        # We need all the points 
        start = 0
        end = patient.num_points
    
    # Parameters for the targets
    p = params["Optimization_targets"]
    measurements = {}


    # !! FIX THIS LATER !!
    if params["synth_data"]:

        synth_output =  params["outdir"] +  "/synth_data.h5"
        num_points = SYNTH_PASSIVE_FILLING + NSYNTH_POINTS + 1
            
        pressure, volume, strain = load_synth_data(patient.mesh, synth_output, num_points, params["use_deintegrated_strains"])
        
            
    else:

        ## Pressure
        
        # We need the pressure as a BC
        pressure = np.array(patient.pressure)
   
        # Compute offsets
        # Choose the pressure at the beginning as reference pressure
        reference_pressure = pressure[0] 
        logger.info("LV Pressure offset = {} kPa".format(reference_pressure))

        #Here the issue is that we do not have a stress free reference mesh. 
        #The reference mesh we use is already loaded with a certain
        #amount of pressure, which we remove.
        pressure = np.subtract(pressure,reference_pressure)
        
        measurements["pressure"] = pressure[start:end]

        if patient.mesh_type() == "biv":
            rv_pressure = np.array(patient.RVP)
            reference_pressure = rv_pressure[0]
            logger.info("RV Pressure offset = {} kPa".format(reference_pressure))
            
            rv_pressure = np.subtract(rv_pressure, reference_pressure)
            measurements["rv_pressure"] = rv_pressure[start:end]
            
        
        
        ## Volume
        if p["volume"]:
            # Calculate difference bwtween calculated volume, and volume given from echo
            volume_offset = get_volume_offset(patient)
            logger.info("LV Volume offset = {} cm3".format(volume_offset))

            
            # Subtract this offset from the volume data
            volume = np.subtract(patient.volume,volume_offset)

            measurements["volume"] = volume[start:end]


        if p["rv_volume"]:
            # Calculate difference bwtween calculated volume, and volume given from echo
            volume_offset = get_volume_offset(patient, "rv")
            logger.info("RV Volume offset = {} cm3".format(volume_offset))

            # Subtract this offset from the volume data
            volume = np.subtract(patient.RVV ,volume_offset)

            measurements["rv_volume"] = volume[start:end]
                

        if p["regional_strain"]:

            strain = {}
            for region in patient.strain.keys():
                strain[region] = patient.strain[region][start:end]
                
            measurements["regional_strain"] = strain
    

    return measurements

def get_volume_offset(patient, chamber = "lv"):
    N = FacetNormal(patient.mesh)

    if chamber == "lv":
    
        if patient.mesh_type() == "biv":
            endo_marker = patient.ENDO_LV
        else:
            endo_marker = patient.ENDO

        volume = patient.volume[0]
        
    else:
        endo_marker = patient.ENDO_RV
        volume = patient.RVV[0]
        
    ds = Measure("exterior_facet",
                 subdomain_data = patient.facets_markers,
                 domain = patient.mesh)(endo_marker)
    
    X = SpatialCoordinate(patient.mesh)
    
    # Divide by 1000 to get the volume in ml
    vol = assemble((-1.0/3.0)*dot(X,N)*ds)
    
    return volume - vol

def setup_simulation(params, patient):

    # Load measurements
    measurements = get_measurements(params, patient)
    solver_parameters, pressure, controls = make_solver_params(params, patient, measurements)
   
    return measurements, solver_parameters, pressure, controls


class MyReducedFunctional(ReducedFunctional):
    def __init__(self, for_run, paramvec, scale = 1.0):
        self.for_run = for_run
        self.paramvec = paramvec
        self.first_call = True
        self.scale = scale
        self.nr_crashes = 0
        self.iter = 0
        self.nr_der_calls = 0
        self.func_values_lst = []
        self.controls_lst = []
        self.forward_times = []
        self.backward_times = []
        self.initial_paramvec = gather_broadcast(paramvec.vector().array())


    def __call__(self, value):
        
        adj_reset()
        self.iter += 1
        paramvec_new = Function(self.paramvec.function_space(), name = "new control")

        if isinstance(value, (Function, RegionalParameter, MixedParameter)):
            paramvec_new.assign(value)
        elif isinstance(value, float) or isinstance(value, int):
            assign_to_vector(paramvec_new.vector(), np.array([value]))
        elif isinstance(value, enlisting.Enlisted):
            val_delisted = delist(value,self.controls)
            paramvec_new.assign(val_delisted)
            
        else:
            assign_to_vector(paramvec_new.vector(), gather_broadcast(value))

    
        logger.debug(Text.yellow("Start annotating"))
        # arr = gather_broadcast(paramvec_new.vector().array())
        # logger.info("Try value {} (mean)".format(arr.mean()))
        parameters["adjoint"]["stop_annotating"] = False

        logger.setLevel(WARNING)
        t = Timer("Forward run")
        t.start()
        self.for_res, crash= self.for_run(paramvec_new, True)
        for_time = t.stop()
        self.forward_times.append(for_time)
        logger.setLevel(INFO)

        if self.first_call:
            # Store initial results 
            self.ini_for_res = self.for_res
            self.first_call = False

            # Some printing
            logger.info(print_head(self.for_res))
	 
        
        control = Control(self.paramvec)
            

        ReducedFunctional.__init__(self, Functional(self.for_res["total_functional"]), control)

        if crash:
            # This exection is thrown if the solver uses more than x steps.
            # The solver is stuck, return a large value so it does not get stuck again
            logger.warning(Text.red("Iteration limit exceeded. Return a large value of the functional"))
            # Return a big value, and make sure to increment the big value so the 
            # the next big value is different from the current one. 
            func_value = np.inf
            self.nr_crashes += 1
    
        else:
            func_value = self.for_res["func_value"]

        
        self.func_values_lst.append(func_value*self.scale)
        self.controls_lst.append(Vector(paramvec_new.vector()))

        
        logger.debug(Text.yellow("Stop annotating"))
        parameters["adjoint"]["stop_annotating"] = True

        # Some printing
        logger.info(print_line(self.for_res, self.iter))

        return self.scale*func_value

    def derivative(self, *args, **kwargs):
        self.nr_der_calls += 1
        import math

        t = Timer("Backward run")
        t.start()
        
        out = ReducedFunctional.derivative(self, forget = False)
        
        back_time = t.stop()
        self.backward_times.append(back_time)
        
        for num in out[0].vector().array():
            if math.isnan(num):
                raise Exception("NaN in adjoint gradient calculation.")

        gathered_out = gather_broadcast(out[0].vector().array())
        
        return self.scale*gathered_out


class RegionalParameter(dolfin.Function):
    def __init__(self, meshfunction):

        assert isinstance(meshfunction, MeshFunctionSizet), \
            "Invalid meshfunction for regional gamma"
        
        mesh = meshfunction.mesh()

        self._values = set(gather_broadcast(meshfunction.array()))
        self._nvalues = len(self._values)
        
        
        V  = dolfin.VectorFunctionSpace(mesh, "R", 0, dim = self._nvalues)
        
        dolfin.Function.__init__(self, V)
        self._meshfunction = meshfunction

        # Functionspace for the indicator functions
        self._IndSpace = dolfin.FunctionSpace(mesh, "DG", 0)
       
        # Make indicator functions
        self._ind_functions = []
        for v in self._values:
            self._ind_functions.append(self._make_indicator_function(v))

    def get_ind_space(self):
        return self._IndSpace
    
    def get_values(self):
        return self._values
    
    def get_function(self):
        """
        Return linear combination of coefficents
        and basis functions

        :returns: A function with parameter values at each segment
                  specified by the meshfunction
        :rtype:  :py:class`dolfin.Function             
             
        """
        return self._sum()

    def _make_indicator_function(self, marker):
        dm = self._IndSpace.dofmap()
        cell_dofs = [dm.cell_dofs(i) for i in
                     np.where(self._meshfunction.array() == marker)[0]]
        dofs = np.unique(np.array(cell_dofs))
        
        f = dolfin.Function(self._IndSpace)
        f.vector()[dofs] = 1.0    
        return f  

    def _sum(self):
        coeffs = dolfin.split(self)
        fun = coeffs[0]*self._ind_functions[0]

        for c,f in zip(coeffs[1:], self._ind_functions[1:]):
            fun += c*f

        return fun


class MixedParameter(dolfin.Function):
    def __init__(self, fun, n, name = "material_parameters"):
        """
        Initialize Mixed parameter.

        This will instanciate a function in a dolfin.MixedFunctionSpace
        consiting of `n` subspaces of the same type as `fun`.
        This is of course easy for the case when `fun` is a normal
        dolfin function, but in the case of a `RegionalParameter` it
        is not that straight forward. 
        This class handles this case as well. 

        

        :param fun: The type of you want to make a du
        :type fun: (:py:class:`dolfin.Function`)
        :param int n: number of subspaces 
        :param str name: Name of the function

        .. todo::
        
           Implement support for MixedParameter with different
           types of subspaces, e.g [RegionalParamter, R_0, CG_1]

        """
    
        msg = "Please provide a dolin function as argument to MixedParameter"
        assert isinstance(fun, (dolfin.Function, RegionalParameter)), msg

        if isinstance(fun, RegionalParameter):
            raise NotImplementedError


        # We can just make a usual mixed function space
        # with n copies of the original one
        V  = fun.function_space()
        W = dolfin.MixedFunctionSpace([V]*n)
        
        dolfin.Function.__init__(self, W, name = name)
        
        # Create a function assigner
        self.function_assigner \
            =  [dolfin.FunctionAssigner(W.sub(i), V) for i in range(n)]

        # Store the original function space
        self.basespace = V

        if isinstance(fun, RegionalParameter):
            self._meshfunction = fun._meshfunction

        
            
    def assign_sub(self, f, i):
        """
        Assign subfunction

        :param f: The function you want to assign
        :param int i: The subspace number

        """
        f_ = Function(self.basespace)
        f_.assign(f)
        self.function_assigner[i].assign(self.split()[i], f_)
        
            


class BaseExpression(Expression):
    """
    A class for assigning boundary condition according to segmented surfaces
    Since the base is located at x = a (usually a=0), two classes must be set: 
    One for the y-direction and one for the z-direction

    Point on the endocardium and epicardium is given and the
    points on the mesh base is set accordingly.
    Points that lie on the base but not on the epi- or endoring
    will be given a zero value.
    """
    def __init__(self, mesh_verts, seg_verts, sub, it, name):
        """
        
        *Arguments*
          mesh: (dolfin.mesh)
            The mesh

          u: (dolfin.GenericFunction)
            Initial displacement

          mesh_verts (numpy.ndarray or list)
            Point of endocardial base from mesh

          seg_verts (numpy.ndarray or list)
            Point of endocardial base from segmentation

          sub (str)
            Either "y" or "z". The displacement in this direction is returned

          it (dolfin.Expression)
            Can be used to incrment the direclet bc

        """ 
        assert sub in ["y", "z"]
        self._mesh_verts = np.array(mesh_verts)
        self._all_seg_verts = np.array(seg_verts)
        self.point = 0
        self.npoints = len(seg_verts)-1
        
        self._seg_verts = self._all_seg_verts[0]
  
        self._sub = sub
        self._it = it
        self.rename(name, name)

        
    def next(self):
        self._it.t = 0
        self.point += 1
        self._seg_verts = self._all_seg_verts[self.point]
     
    def reset(self):
        self.point = 0
        self._it.t = 0
    

    def eval(self, value, x):

        # Check if given coordinate is in the endoring vertices
        # and find the cooresponding index
        d = [np.where(x[i] == self._mesh_verts.T[i])[0] for i in range(3)]
        d_intersect = set.intersection(*map(set,d))
        assert len(d_intersect) < 2
        if len(d_intersect) == 1:
          
            idx = d_intersect.pop()
            
            prev_seg_verts = self._all_seg_verts[self.point-1] 

            # Return the displacement in the given direction
            # Iterated starting from the previous displacemet to the current one
            if self._sub == "y":
                u_prev = self._mesh_verts[idx][1] - prev_seg_verts[idx][1]
                u_current = self._mesh_verts[idx][1] - self._seg_verts[idx][1]
                # value[0] = u_prev + self._it.t*(u_current - u_prev)
            else: # sub == "z"
                u_prev = self._mesh_verts[idx][2] - prev_seg_verts[idx][2]
                u_current = self._mesh_verts[idx][2] - self._seg_verts[idx][2]

            val = u_prev + self._it.t*(u_current - u_prev)
            value[0] = val
            
        else:
            value[0] = 0
          

class VertexDomain(SubDomain):
    """
    A subdomain defined in terms of
    a given set of coordinates.
    A point that is close to the given coordinates
    within a given tolerance will be marked as inside 
    the domain.
    """
    def __init__(self, coords, tol=1e-4):
        """
        *Arguments*
          coords (list)
            List of coordinates for vertices in reference geometry
            defining this domains

          tol (float)
            Tolerance for how close a pointa should be to the given coordinates
            to be marked as inside the domain
        """
        self.coords = np.array(coords)
        self.tol = tol
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        
        if np.all([np.any(abs(x[i] - self.coords.T[i]) < self.tol) for i in range(3)]):
            return True
        
        return False
