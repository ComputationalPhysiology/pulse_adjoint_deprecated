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
from utils import Object, Text
from adjoint_contraction_args import *
from numpy_mpi import *



def setup_adjoint_contraction_parameters():

    params = setup_application_parameters()

    # Patient parameters
    patient_parameters = setup_patient_parameters()
    params.add(patient_parameters)

    # Optimization parameters
    opt_parameters = setup_optimization_parameters()
    params.add(opt_parameters)

    opttarget_parameters = setup_optimizationtarget_parameters()
    params.add(opttarget_parameters)
    
    return params
    
    
def setup_solver_parameters():
    # from dolfin.cpp.fem import NonlinearVariationalSolver
    #NonlinearVariationalSolver.default_parameters()
    solver_parameters = {"snes_solver":{}}

    solver_parameters["nonlinear_solver"] = NONLINSOLVER
    solver_parameters["snes_solver"]["method"] = SNES_SOLVER_METHOD
    solver_parameters["snes_solver"]["maximum_iterations"] = SNES_SOLVER_MAXITR
    solver_parameters["snes_solver"]["absolute_tolerance"] = SNES_SOLVER_ABSTOL 
    solver_parameters["snes_solver"]["linear_solver"] = SNES_SOLVER_LINSOLVER
    
    

    return solver_parameters
    
   

def setup_general_parameters():

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
    params = Parameters("Patient_parameters")
    params.add("patient", DEFAULT_PATIENT)
    params.add("patient_type", DEFAULT_PATIENT_TYPE)
    params.add("weight_rule", DEFAULT_WEIGHT_RULE, WEIGHT_RULES)
    params.add("weight_direction", DEFAULT_WEIGHT_DIRECTION, WEIGHT_DIRECTIONS) 
    params.add("resolution", "low_res")
    params.add("fiber_angle_epi", 50)
    params.add("fiber_angle_endo", 40)
    params.add("mesh_type", "lv", ["lv", "biv"])
    params.add("include_sheets", True)

    return params

def setup_optimizationtarget_parameters():

    params = Parameters("Optimization_targets")
    params.add("volume", True)
    params.add("regional_strain", True)
    params.add("full_strain", False)
    params.add("GL_strain", False)
    params.add("displacement", False)
    return params
    
def setup_application_parameters():

    params = Parameters("Application_parmeteres")

    ## Output ##
    
    # Location of output
    params.add("sim_file", DEFAULT_SIMULATION_FILE)
    params.add("outdir", os.path.dirname(DEFAULT_SIMULATION_FILE))

    ## Parameters ##
    
    ## Weight of strain vs volume match
    # Active phase
    params.add("alpha", ALPHA)
    # Passive phase
    params.add("alpha_matparams", ALPHA_MATPARAMS)

    # Regularization parameter
    params.add("reg_par", REG_PAR)
    
    # Spring constant at base (Note: works one for base_bc = fix_x)
    params.add("base_spring_k", BASE_K)

    # Material parameters
    material_parameters = Parameters("Material_parameters")
    material_parameters.add("a", INITIAL_MATPARAMS[0])
    material_parameters.add("a_f", INITIAL_MATPARAMS[1])
    material_parameters.add("b", INITIAL_MATPARAMS[2])
    material_parameters.add("b_f", INITIAL_MATPARAMS[3])
    params.add(material_parameters)

    # Ratio a/a_f used to constrain the passive optimization
    # if None then no constraint are put on the optimization
    params.add("linear_matparams_ratio", 0.0)
    

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
    
    # Space for active parameter
    params.add("gamma_space", "regional", ["CG_1", "R_0", "regional"])

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


    return params

def setup_optimization_parameters():
    # Parameters for the Scipy Optimization
    params = Parameters("Optimization_parmeteres")
    params.add("method", DEFAULT_OPTIMIZATION_METHOD)
    params.add("active_opt_tol", OPTIMIZATION_TOLERANCE_GAMMA)
    params.add("active_maxiter", OPTIMIZATION_MAXITER_GAMMA)
    params.add("passive_opt_tol", OPTIMIZATION_TOLERANCE_MATPARAMS)
    params.add("passive_maxiter", OPTIMIZATION_MAXITER_MATPARAMS)
    params.add("scale", SCALE)
    params.add("gamma_max", MAX_GAMMA)
    params.add("matparams_min", 0.1)
    params.add("matparams_max", 50.0)
    params.add("fix_a", False)
    params.add("fix_a_f", False)
    params.add("fix_b", False)
    params.add("fix_b_f", False)
   
    
    params.add("disp", False)

    return params


def initialize_patient_data(patient_parameters, synth_data):

    
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

    if hasattr(patient, "e_s"):
        fields = [patient.e_f, patient.e_s, patient.e_sn]
    else:
        fields = [patient.e_f]

    local_basis = [patient.e_circ, patient.e_rad, patient.e_long]
        
    save_geometry_to_h5(patient.mesh, sim_file, "", patient.markers,
                            fields, local_basis)
    
    
    # with HDF5File(mpi_comm_world(), sim_file, file_format) as h5file:
    #     h5file.write(patient.mesh, 'geometry/mesh')

        
    #     fgroup = "microstructure"
    #     names = []
    #     for field in [patient.e_f, patient.e_s, patient.e_sn]:
    #         name = "{}_{}".format(str(field), field.label())
    #         fsubgroup = "{}/{}".format(fgroup, name)
    #         h5file.write(field, fsubgroup)
    #         h5file.attributes(fsubgroup)['name'] = field.name()
    #         names.append(name)

    #     elm = field.function_space().ufl_element()
    #     family, degree = elm.family(), elm.degree()
    #     fspace = '{}_{}'.format(family, degree)
    #     h5file.attributes(fgroup)['space'] = fspace
    #     h5file.attributes(fgroup)['names'] = ":".join(names)

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
    
    # Material parameters

    # If we want to estimate material parameters, use the materal parameters
    # from the parameters
    if params["phase"] in [PHASES[0], PHASES[2]]:
        
        material_parameters = params["Material_parameters"]
        paramvec = Function(VectorFunctionSpace(patient.mesh, "R", 0, dim = 4), name = "matparam vector")
        assign_to_vector(paramvec.vector(), np.array(material_parameters.values()))
        

    # Otherwise load the parameters from the result file  
    else:

        # Open simulation file
        with HDF5File(mpi_comm_world(), params["sim_file"], 'r') as h5file:
        
                # Get material parameter from passive phase file
                paramvec = Function(VectorFunctionSpace(patient.mesh, "R", 0, dim = 4), name = "matparam vector")
                h5file.read(paramvec, PASSIVE_INFLATION_GROUP.format(params["alpha_matparams"]) +
                            "/parameters/optimal_material_parameters_function")

    

        
    a,a_f,b,b_f = split(paramvec)

    # Contraction parameter
    if params["gamma_space"] == "regional":
        gamma = RegionalGamma(patient.strain_markers)
    else:
        gamma_family, gamma_degree = params["gamma_space"].split("_")
        gamma_space = FunctionSpace(patient.mesh, gamma_family, int(gamma_degree))

        gamma = Function(gamma_space, name = 'activation parameter')


    strain_weights = patient.strain_weights
    
    strain_weights_deintegrated = patient.strain_weights_deintegrated \
      if params["use_deintegrated_strains"] else None
    
        

    p_lv = Expression("t", t = measurements.pressure[0], name = "LV_endo_pressure")
    N = FacetNormal(patient.mesh)

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
        

        if 0:
            # Plot the points on the endoring
            sub_domains = MeshFunction("size_t", patient.mesh, 0)
            sub_domains.set_all(0)
            endoring.mark(sub_domains, 1)
            plot(sub_domains, interactive=True)
            
            base_it.t = 1.0
            V = FunctionSpace(patient.mesh, "CG", 1)
            VV = VectorFunctionSpace(patient.mesh, "CG", 1, dim = 3)

            funy = interpolate(base_bc_y, V)
            funz = interpolate(base_bc_z, V)

            u_dir_weak = Function(VV)
    
            fa = [FunctionAssigner(VV.sub(i+1), V) for i in range(2)]
            fa[0].assign(u_dir_weak.split()[1], funy)
            fa[1].assign(u_dir_weak.split()[2], funz)
        
            plot(funy, title = "y")
            plot(funz, title = "z")
            plot(u_dir_weak, title = "y+z")
            interactive()
            exit()

            
            
   
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


    from material import HolzapfelOgden
    matparams = {"a":a, "a_f":a_f, "b":b, "b_f":b_f}
    if params["active_model"] == "active_strain_rossi":
        material = HolzapfelOgden(patient.e_f, gamma, matparams, params["active_model"],
                                      patient.strain_markers, s0 = patient.e_s, n0 = patient.e_sn)
    else:
        material = HolzapfelOgden(patient.e_f, gamma, matparams,
                                      params["active_model"], patient.strain_markers)

    crl_basis = (patient.e_circ, patient.e_rad, patient.e_long)
    
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
                               "neumann":[[p_lv, patient.ENDO]],
                               "robin": robin_bc},
                         "solve":setup_solver_parameters()}


    pararr = gather_broadcast(paramvec.vector().array())
    logger.info("\nParameters")
    logger.info("\ta     = {:.3f}".format(pararr[0]))
    logger.info("\ta_f   = {:.3f}".format(pararr[1]))
    logger.info("\tb     = {:.3f}".format(pararr[2]))
    logger.info("\tb_f   = {:.3f}".format(pararr[3]))
    logger.info('\talpha = {}'.format(params["alpha"]))
    logger.info('\talpha_matparams = {}'.format(params["alpha_matparams"]))
    logger.info('\treg_par = {}\n'.format(params["reg_par"]))


    if params["phase"] in [PHASES[0], PHASES[2]]:
        return solver_parameters, p_lv, paramvec
    elif params["phase"] == PHASES[1]:
        return solver_parameters, p_lv, gamma
    else:
        return solver_parameters, p_lv

def get_measurements(params, patient):

    # FIXME
    if params["synth_data"]:

        # if hasattr(patient, "datafile"):
        #     synth_output = patient.datafile
        #     pressure, volume, strain = load_synth_data(patient.mesh, synth_output, patient.num_points, params["use_deintegrated_strains"])
        # else:
        synth_output =  params["outdir"] +  "/synth_data.h5"
        num_points = SYNTH_PASSIVE_FILLING + NSYNTH_POINTS + 1
            
        pressure, volume, strain = load_synth_data(patient.mesh, synth_output, num_points, params["use_deintegrated_strains"])
        
            
    else:
        pressure = np.array(patient.pressure)
        # Compute offsets

        # Calculate difference bwtween calculated volume, and volume given from echo
        volume_offset = get_volume_offset(patient)
        logger.info("Volume offset = {} cm3".format(volume_offset))

        # Subtract this offset from the volume data
        volume = np.subtract(patient.volume,volume_offset)

        #Convert pressure to centipascal (the mesh is in cm)
        # pressure = np.multiply(KPA_TO_CPA, pressure)

        # Choose the pressure at the beginning as reference pressure
        reference_pressure = pressure[0] 
        logger.info("Pressure offset = {} kPa".format(reference_pressure))

        #Here the issue is that we do not have a stress free reference mesh. 
        #The reference mesh we use is already loaded with a certain amount of pressure, which we remove.    
        pressure = np.subtract(pressure,reference_pressure)

        if params["use_deintegrated_strains"]:
            patient.load_deintegrated_strains(STRAIN_FIELDS_PATH)
            strain = (patient.strain, patient.strain_deintegrated)

        else:
            strain = patient.strain



    if params["phase"] == PHASES[0]: #Passive inflation
        # We need just the points from the passive phase
        start = 0
        end = patient.passive_filling_duration

    elif params["phase"] == PHASES[1]: #Scalar contraction
        # We need just the points from the active phase
        start = patient.passive_filling_duration -1
        end = len(pressure)
      
    else:
        # We need all the points 
        start = 0
        end = len(pressure)
    
    measurements = Object()
    # Volume
    measurements.volume = volume[start:end]
    
    # Pressure
    measurements.pressure = pressure[start:end]

    # Endoring vertex coordinates from segementation
    measurements.seg_verts = None if not hasattr(patient, 'seg_verts') else patient.seg_verts[start:end]
    
    # Strain 
    if  params["use_deintegrated_strains"]:
        strain, strain_deintegrated = strain
        measurements.strain_deintegrated = strain_deintegrated[start:end] 
    else:
        measurements.strain_deintegrated = None


    strains = {}
    for region in STRAIN_REGION_NUMS:
        strains[region] = strain[region][start:end]
    measurements.strain = strains
    

    return measurements

def get_volume_offset(patient):
    N = FacetNormal(patient.mesh)
    ds = Measure("exterior_facet", subdomain_data = patient.facets_markers, domain = patient.mesh)(patient.ENDO)
    X = SpatialCoordinate(patient.mesh)
    
    # Divide by 1000 to get the volume in ml
    vol = assemble((-1.0/3.0)*dot(X,N)*ds)
    return patient.volume[0] - vol

def setup_simulation(params, patient):
    
    # Load measurements
    measurements = get_measurements(params, patient)
    solver_parameters, p_lv, controls = make_solver_params(params, patient, measurements)

    return measurements, solver_parameters, p_lv, controls


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
        self.initial_paramvec = gather_broadcast(paramvec.vector().array())

    def __call__(self, value):
        adj_reset()
        self.iter += 1

        paramvec_new = Function(self.paramvec.function_space(), name = "new control")

        if isinstance(value, Function) or isinstance(value, RegionalGamma):
            paramvec_new.assign(value)
        else:
            assign_to_vector(paramvec_new.vector(), value)

    
        logger.debug(Text.yellow("Start annotating"))
        parameters["adjoint"]["stop_annotating"] = False

        logger.setLevel(WARNING)
        self.for_res, crash= self.for_run(paramvec_new, True)
        logger.setLevel(INFO)

        if self.first_call:
            # Store initial results 
            self.ini_for_res = self.for_res
            self.first_call = False
            logger.info("Iter\tI_tot\t\tI_vol\t\tI_strain\tI_reg")
	 
        
        control = Control(self.paramvec)
            

        ReducedFunctional.__init__(self, Functional(self.for_res.total_functional), control)

        if crash:
            # This exection is thrown if the solver uses more than x steps.
            # The solver is stuck, return a large value so it does not get stuck again
            logger.warning(Text.red("Iteration limit exceeded. Return a large value of the functional"))
            # Return a big value, and make sure to increment the big value so the 
            # the next big value is different from the current one. 
            func_value = np.inf
            self.nr_crashes += 1
    
        else:
            func_value = self.for_res.func_value

        self.func_values_lst.append(func_value)
        self.controls_lst.append(Vector(paramvec_new.vector()))
        
        logger.debug(Text.yellow("Stop annotating"))
        parameters["adjoint"]["stop_annotating"] = True

        logger.info("{}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}".format(self.iter, 
                                                               func_value, 
                                                               self.for_res.func_value_volume, 
                                                               self.for_res.func_value_strain, 
                                                               self.for_res.gamma_gradient))

        return self.scale*func_value

    def derivative(self, *args, **kwargs):
        self.nr_der_calls += 1
        import math
        out = ReducedFunctional.derivative(self, forget = False)
        for num in out[0].vector().array():
            if math.isnan(num):
                raise Exception("NaN in adjoint gradient calculation.")

        gathered_out = gather_broadcast(out[0].vector().array())
        
        return self.scale*gathered_out



class RealValueProjector(object):
    """
    Projects onto a real valued function in order to force dolfin-adjoint to
    create a recording.
    """
    def __init__(self, u,v, mesh_vol):
        self.u_trial = u
        self.v_test = v
        self.mesh_vol = mesh_vol
    
        
    def project(self, expr, measure, real_function, mesh_vol_divide = True):

        if mesh_vol_divide:
            solve((self.u_trial*self.v_test/self.mesh_vol)*dx == \
                  self.v_test*expr*measure,real_function)
            
        else:
            solve((self.u_trial*self.v_test)*dx == \
              self.v_test*expr*measure,real_function)

        return real_function




class RegionalGamma(dolfin.Function):
    def __init__(self, meshfunction):
        
        mesh = meshfunction.mesh()
        
        V  = dolfin.VectorFunctionSpace(mesh, "R", 0, dim = 17)
        
        dolfin.Function.__init__(self, V)
        self._meshfunction = meshfunction

        # Functionspace for the indicator functions
        self._IndSpace = dolfin.FunctionSpace(mesh, "DG", 0)
       
        # Make indicator functions
        self._ind_functions = []
        for i in range(1,18):
            self._ind_functions.append(self._make_indicator_function(i))

    def get_function(self):
        """
        Return linear combination of coefficents
        and basis functions

        *Returns*
           fun (dolfin.Function)
             A function with gamma values at each segment
             
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
