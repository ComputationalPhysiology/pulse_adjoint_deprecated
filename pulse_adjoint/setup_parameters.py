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
from adjoint_contraction_args import *

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

    unload_params = setup_unloading_parameters()
    params.add(unload_params)
        
    check_parameters(params)
    
    return params
    
    
def setup_solver_parameters():
    """
    Have a look at `dolfin.NonlinearVariationalSolver.default_parameters`
    for options

    """
    solver = "snes"
    solver_str = "{}_solver".format(solver)
    # solver_parameters = {"snes_solver":{}}
    solver_parameters = {solver_str:{}}

    solver_parameters["nonlinear_solver"] = solver
    solver_parameters[solver_str]["method"] = "newtontr"
    solver_parameters[solver_str]["maximum_iterations"] = 50
    solver_parameters[solver_str]["absolute_tolerance"] = 1.0e-5
    solver_parameters[solver_str]["linear_solver"] = "lu"
    # solver_parameters[solver_str]["convergence_criterion"] = "incremental"
    # solver_parameters[solver_str]['relaxation_parameter'] = 0.5
    # set_log_active(True)
    # set_log_level(INFO)
    

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
    params.add("mesh_group", "")
    
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
    # Store the results in the file within a folder
    params.add("h5group", "")
    params.add("outdir", os.path.dirname(params["sim_file"]))

    ## Parameters ##
    
    # Spring constant at base (Note: works one for base_bc = fix_x)
    params.add("base_spring_k", 1.0)

    # Spring constatnt at pericardium (if zero - divergence free)
    params.add("pericardium_spring", 0.0)

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
    params.add("phase", PHASES[0])

    # Iteration for active phase
    params.add("active_contraction_iteration_number", 0)
    

    ## Additional setup ##

    # Do you want to find the unloaded geometry and use that?
    params.add("unload", False)
    
    # For passive optimization, include all passive points ('all')
    # or only the final point ('-1'), or specific point ('point')
    params.add("passive_weights", "all")
    
    # Update weights so that the initial value of the functional is 0.1
    params.add("adaptive_weights", True)
    
    # Space for active parameter
    params.add("gamma_space", "CG_1")#, ["CG_1", "R_0", "regional"])
    
    # If you want to use pointswise strains as input (only synthetic)
    params.add("use_deintegrated_strains", False)

    # If you want to optimize passive parameters
    params.add("optimize_matparams", True)

    # Normalization factor for active contraction
    # For default values see material module
    params.add("T_ref", 0.0)

    # If you want to use a zero initial guess for gamma (False),
    # or use gamma from previous iteration as initial guess (True)
    params.add("initial_guess", "previous", ["previous", "zero", "smooth"])

    # Log level
    params.add("log_level", logging.INFO)
    # If False turn of logging of the forward model during functional evaluation
    params.add("verbose", False)

    # If you optimize against strain which reference geometry should be used
    # to compute the strains.  "0" is the starting geometry, "ED" is the end-diastolic
    # geometry, while if you are using unloading, you can also use that geometry as referece. 
    params.add("strain_reference", "0", ["0", "ED", "unloaded"])
    
    # Relaxation parameters. If smaller than one, the step size
    # in the direction will be smaller, and perhaps avoid the solver
    # to crash.
    params.add("passive_relax", 0.1)
    params.add("active_relax", 0.001)


    # When computing the volume/strain, do you want to the project or  interpolate
    # the diplacement onto a CG 1 space, or do you want to keep the original
    # displacement (default CG2)
    params.add("volume_approx", "project", ["project", "interpolate", "original"])
    params.add("strain_approx", "original", ["project", "interpolate", "original"])

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
    params = Parameters("Optimization_parameters")
    params.add("opt_type", "scipy_slsqp")
    params.add("method_1d", "bounded")
    params.add("active_opt_tol", 1e-10)
    params.add("active_maxiter", 100)
    params.add("passive_opt_tol", 1e-10)
    params.add("passive_maxiter", 30)
    params.add("scale", 1.0)
    
    params.add("gamma_min", 0.0)
    params.add("gamma_max", 0.4)
    
    params.add("matparams_min", 1.0)
    params.add("matparams_max", 50.0)
    params.add("fix_a", False)
    params.add("fix_a_f", True)
    params.add("fix_b", True)
    params.add("fix_b_f", True)

    params.add("soft_tol", 1e-6)
    params.add("soft_tol_rel", 0.1)

    params.add("adapt_scale", True)
    params.add("disp", False)

    return params

def setup_unloading_parameters():
    """
    Parameters for coupled unloading/material parameter
    estimation. 

    For info about the different parameters, 
    see the unloading module. 
    """

    params = Parameters("Unloading_parameters")

    params.add("method", "hybrid", ["hybrid", "fixed_point", "raghavan"])
    params.add("tol", 0.05)
    params.add("maxiter", 5)

    unload_options = Parameters("unload_options")
    unload_options.add("maxiter", 10)
    unload_options.add("tol", 0.01)
    unload_options.add("regen_fibers", False)
    
    params.add(unload_options)

    return params
