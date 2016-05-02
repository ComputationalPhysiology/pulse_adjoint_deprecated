from dolfin import *
from dolfin_adjoint import *
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
    
    dolfin.set_log_active(True)
    dolfin.set_log_level(ERROR)


def setup_patient_parameters():
    params = Parameters("Patient_parameters")
    params.add("patient", DEFAULT_PATIENT)
    params.add("patient_type", DEFAULT_PATIENT_TYPE)
    params.add("weight_rule", DEFAULT_WEIGHT_RULE, WEIGHT_RULES)
    params.add("weight_direction", DEFAULT_WEIGHT_DIRECTION, WEIGHT_DIRECTIONS) 
    params.add("resolution", "med_res")
    params.add("fiber_angle_epi", 50)
    params.add("fiber_angle_endo", 40)
    params.add("mesh_type", "lv", ["lv", "biv"])

    return params

def setup_application_parameters():

    params = Parameters("Application_parmeteres")
    params.add("sim_file", DEFAULT_SIMULATION_FILE)
    
    params.add("outdir", os.path.dirname(DEFAULT_SIMULATION_FILE))
    params.add("alpha", ALPHA)
    params.add("base_spring_k", BASE_K)
    params.add("reg_par", REG_PAR)
    params.add("gamma_space", "CG_1", ["CG_1", "R_0"])
    params.add("state_space", "P_2:P_1")
    params.add("compressibility", "incompressible", ["incompressible", 
                                                     "stabalized_incompressible", 
                                                     "penalty", "hu_washizu"])
    params.add("incompressibility_penalty", 10.0)
    params.add("use_deintegrated_strains", False)
    params.add("optimize_matparams", True)
    params.add("nonzero_initial_guess", True)

    params.add("synth_data", False)
    params.add("noise", False)
    
    # Set material parameter estimation as default
    params.add("phase", PHASES[0], PHASES)
    params.add("alpha_matparams", ALPHA_MATPARAMS)
    params.add("active_contraction_iteration_number", 0)

    material_parameters = Parameters("Material_parameters")
    material_parameters.add("a", INITIAL_MATPARAMS[0])
    material_parameters.add("a_f", INITIAL_MATPARAMS[1])
    material_parameters.add("b", INITIAL_MATPARAMS[2])
    material_parameters.add("b_f", INITIAL_MATPARAMS[3])
    params.add(material_parameters)



    return params

def setup_optimization_parameters():
    # Parameters for the Scipy Optimization
    params = Parameters("Optimization_parmeteres")
    params.add("method", OPTIMIZATION_METHOD)
    params.add("active_opt_tol", OPTIMIZATION_TOLERANCE_GAMMA)
    params.add("active_maxiter", OPTIMIZATION_MAXITER_GAMMA)
    params.add("passive_opt_tol", OPTIMIZATION_TOLERANCE_MATPARAMS)
    params.add("passive_maxiter", OPTIMIZATION_MAXITER_MATPARAMS)
    params.add("scale", SCALE)
    params.add("gamma_max", MAX_GAMMA)
    params.add("matparams_min", 0.1)
    params.add("matparams_max", 50.0)
    params.add("disp", False)

    return params


def initialize_patient_data(patient_parameters, synth_data):

    logger.info("Initialize patient data")
    from patient_data import Patient
    
    patient = Patient(**patient_parameters)
    
    # if args_full.use_deintegrated_strains:
        # patient.load_deintegrated_strains(STRAIN_FIELDS_PATH)

    if synth_data:
        patient.passive_filling_duration = SYNTH_PASSIVE_FILLING
        patient.num_contract_points =  NSYNTH_POINTS + 1
        patient.num_points = SYNTH_PASSIVE_FILLING + NSYNTH_POINTS + 1

    return patient

def save_patient_data_to_simfile(patient, sim_file):

    file_format = "a" if os.path.isfile(sim_file) else "w"
    
    with HDF5File(mpi_comm_world(), sim_file, file_format) as h5file:
        h5file.write(patient.mesh, 'geometry/mesh')
        fgroup = "microstructure"
        names = []
        for field in [patient.e_f, patient.e_s, patient.e_sn]:
            name = "{}_{}".format(str(field), field.label())
            fsubgroup = "{}/{}".format(fgroup, name)
            h5file.write(field, fsubgroup)
            h5file.attributes(fsubgroup)['name'] = field.name()
            names.append(name)

        elm = field.function_space().ufl_element()
        family, degree = elm.family(), elm.degree()
        fspace = '{}_{}'.format(family, degree)
        h5file.attributes(fgroup)['space'] = fspace
        h5file.attributes(fgroup)['names'] = ":".join(names)

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
    gamma_family, gamma_degree = params["gamma_space"].split("_")
    gamma_space = FunctionSpace(patient.mesh, gamma_family, int(gamma_degree))
    gamma = Function(gamma_space, name = 'activation parameter')


    strain_weights = patient.strain_weights
    
    strain_weights_deintegrated = patient.strain_weights_deintegrated \
      if params["use_deintegrated_strains"] else None
    
        

    p_lv = Expression("t", t = measurements.pressure[0])
    
    
    N = FacetNormal(patient.mesh)
    
    def make_dirichlet_bcs(W):
	'''Make Dirichlet boundary conditions where the base is allowed to slide
        in the x = 0 plane.
        '''
        
        
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = DirichletBC(V.sub(0), 0, patient.BASE)
        return [no_base_x_tran_bc]
	
    from lvsolver import HolzapfelOgden

    matparams = {"a":a, "a_f":a_f, "b":b, "b_f":b_f}
    material = HolzapfelOgden(patient.e_f, gamma, matparams, "active_strain")
    
    solver_parameters = {"mesh": patient.mesh,
                         "facet_function": patient.facets_markers,
                         "facet_normal": N,
                         "mesh_function": patient.strain_markers,
                         "strain_weights": strain_weights, 
                         "strain_weights_deintegrated": strain_weights_deintegrated,
                         "state_space": "P_2:P_1",
                         "compressibility":{"type": params["compressibility"],
                                            "lambda": params["incompressibility_penalty"]},
                         "material": material,
                         "bc":{"dirichlet": make_dirichlet_bcs,
                               "neumann":[[p_lv, patient.ENDO]],
                               "robin":[[-Constant(params["base_spring_k"], 
                                                   name ="base_spring_constant"), patient.BASE]]},
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

    if params["phase"] == PHASES[0]:
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
        pressure = np.multiply(KPA_TO_CPA, pressure)


        # Choose the pressure at the beginning as reference pressure
        reference_pressure = pressure[0] 
        logger.info("Pressure offset = {} cPa".format(reference_pressure))

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
        self.big_value = 100
        self.nr_crashes = 0
        self.iter = 0
        self.nr_der_calls = 0
        self.initial_paramvec = gather_broadcast(paramvec.vector().array())

    def __call__(self, value):
        adj_reset()
        self.iter += 1


        paramvec_new = Function(self.paramvec.function_space(), name = "new control")

        if isinstance(value, Function):
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
	 
        ReducedFunctional.__init__(self, Functional(self.for_res.total_functional), Control(self.paramvec))

        if crash:
            # This exection is thrown if the solver uses more than x times.
            # The solver is stuck, return a large value so it does not get stuck again
            logger.warning(Text.red("Iteration limit exceeded. Return a large value of the functional"))
            # Return a big value, and make sure to increment the big value so the 
            # the next big value is different from the current one. 
            func_value = self.big_value
            self.big_value += 100
            self.nr_crashes += 1
    
        else:
            func_value = self.for_res.func_value

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

    # def project_volume_diff(self, expr, measure, real_function):
        
    #     solve((self.u_trial*self.v_test)*dx == \
    #           self.v_test*expr*measure,real_function)

    #     return real_function

        
    
