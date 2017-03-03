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
from setup_parameters import *

def update_unloaded_patient(params, patient):

    # Make sure to load the new referece geometry
    from mesh_generation import load_geometry_from_h5
    h5group = "/".join(filter(None, [params["h5group"], "unloaded"]))
    geo = load_geometry_from_h5(params["sim_file"], h5group)
    setattr(patient, "original_geometr", getattr(patient, "mesh"))
    for k, v in geo.__dict__.iteritems():
        if hasattr(patient, k):
            delattr(patient, k)
            
        setattr(patient, k, v)
        
    return patient

    
def initialize_patient_data(patient_parameters, synth_data=False):
    """
    Make an instance of patient from :py:module`patient_data`
    baed on th given parameters

    Parameters
    ----------
    patient_parameters: dict
        the parameters 
    synth_data: bool
        If synthetic data or not

    Returns
    -------
    patient: :py:class`patient_data.Patient`
        A patient instance

    **Example of usage**::
    
      params = setup_patient_parameters()
      patient = initialize_patient_data(params, False)

    """
    
    logger.info("Initialize patient data")
    from patient_data import Patient
    
    patient = Patient(**patient_parameters)

    if synth_data:
        patient.passive_filling_duration = SYNTH_PASSIVE_FILLING
        patient.num_contract_points =  NSYNTH_POINTS + 1
        patient.num_points = SYNTH_PASSIVE_FILLING + NSYNTH_POINTS + 1

    return patient


def check_patient_attributes(patient):
    """
    Check that the object contains the minimum 
    required attributes. 
    """

    msg = "Patient is missing attribute {}"

    # Mesh
    if not hasattr(patient, 'mesh'):
        raise AttributeError(msg.format("mesh"))
    else:
        dim = patient.mesh.topology().dim()



    ## Microstructure 

    # Fibers
    if not hasattr(patient, 'fiber'):

        no_fiber = True
        if hasattr(patient, 'e_f'):
            rename_attribute(patient, 'e_f', 'fiber')
            no_fiber = False
            
        if no_fiber:

            idx_arr = np.where([item.startswith("fiber") \
                                for item in dir(patient)])[0]
            if len(idx) == 0:
                raise AttributeError(msg.format("fiber"))
            else:
                att = dir(patient)[idx_arr[0]]
                rename_attribute(patient, att, 'fiber')

    # Sheets
    if not hasattr(patient, 'sheet') and hasattr(patient, 'e_s'):
        rename_attribute(patient, 'e_s', 'sheet')
    else:
        setattr(patient, 'sheet', None)

    # Cross-sheet
    if not hasattr(patient, 'sheet_normal') and hasattr(patient, 'e_sn'):
        rename_attribute(patient, 'e_sn', 'sheet_normal')
    else:
        setattr(patient, 'sheet_normal', None)


    ## Local basis

    # Circumferential
    if not hasattr(patient, 'circumferential') \
       and hasattr(patient, 'e_circ'):
        rename_attribute(patient, 'e_circ', 'circumferential')

    # Radial
    if not hasattr(patient, 'radial') \
       and hasattr(patient, 'e_rad'):
        rename_attribute(patient, 'e_rad', 'radial')

    # Longitudinal
    if not hasattr(patient, 'longitudinal') \
       and hasattr(patient, 'e_long'):
        rename_attribute(patient, 'e_long', 'longitudinal')
        

        
    ## Markings
        
    # Markers 
    if not hasattr(patient, 'markers'):
        raise AttributeError(msg.format("markers"))
        
    # Facet fuction
    if not hasattr(patient, 'ffun'):

        no_ffun = True 
        if hasattr(patient, 'facets_markers'):
            rename_attribute(patient, 'facets_markers', 'ffun')
            no_ffun = False

        if no_ffun:
            setattr(patient, 'strain_weights',
                    MeshFunction("size_t", mesh, 2, mesh.domains()))

    # Cell markers 
    if dim == 3 and not hasattr(patient, 'sfun'):
        
        no_sfun = True 
        if no_sfun and hasattr(patient, 'strain_markers'):
            rename_attribute(patient, 'strain_markers', 'sfun')
            no_sfun = False

        if no_sfun:
            setattr(patient, 'strain_weights',
                    MeshFunction("size_t", mesh, 3, mesh.domains()))


    ## Other

    # Weigts on strain semgements
    if not hasattr(patient, 'strain_weights'):
        setattr(patient, 'strain_weights', None)

    # Mesh type
    if not hasattr(patient, 'mesh_type'):
        # If markers are according to fiberrules, 
        # rv should be marked with 20
        if 20 in set(patient.ffun.array()):
            setattr(patient, 'mesh_type', lambda : 'biv')
        else:
            setattr(patient, 'mesh_type', lambda : 'lv')

    if not hasattr(patient, 'passive_filling_duration'):
        setattr(patient, 'passive_filling_duration', 1)
                    

def save_patient_data_to_simfile(patient, sim_file):

    from mesh_generation.mesh_utils import save_geometry_to_h5

    fields = []
    for att in ["fiber", "sheet", "sheet_normal"]:
        if hasattr(patient, att):
            fields.append(getattr(patient, att))

    local_basis = []
    for att in ["circumferential", "radial", "longitudinal"]:
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

def make_solver_params(params, patient, measurements = None):

    paramvec, gamma, matparams = make_control(params, patient)
    return make_solver_parameters(params, patient, matparams,
                                  gamma, paramvec, measurements)


def make_solver_parameters(params, patient, matparams,
                           gamma = Constant(0.0),
                           paramvec = None, measurements = None):

     ##  Material model
    from material import HolzapfelOgden
    
    material = HolzapfelOgden(patient.fiber, gamma,
                              matparams,
                              params["active_model"],
                              s0 = patient.sheet,
                              n0 = patient.sheet_normal,
                              T_ref = params["T_ref"])
    
        
    if measurements is None:
        p_lv_ = 0.0
        p_rv_ = 0.0
    else:
        p_lv_  = measurements["pressure"][0]
        if measurements.has_key("rv_pressure"):
            p_rv_ =  measurements["rv_pressure"][0]
            
    # Neumann BC
    neuman_bc = []

    V_real = FunctionSpace(patient.mesh, "R", 0)
    p_lv = Expression("t", t = p_lv_, name = "LV_endo_pressure", element = V_real.ufl_element())
    

    if patient.mesh_type() == "biv":
        p_rv = Expression("t", t = p_rv_, name = "RV_endo_pressure", element = V_real.ufl_element())
        
        neumann_bc = [[p_lv, patient.markers["ENDO_LV"][0]],
                     [p_rv, patient.markers["ENDO_RV"][0]]]

        pressure = {"p_lv":p_lv, "p_rv":p_rv}
    else:
        neumann_bc = [[p_lv, patient.markers["ENDO"][0]]]
        pressure = {"p_lv":p_lv}
    

    


    if params["base_bc"] == "from_seg_base":

        # Direchlet BC at the Base
        try:
            mesh_verts = patient.mesh_verts
            seg_verts = measurements.seg_verts
        except:
            raise ValueError(("No mesh vertices found. Fix base "+
                              "is the only applicable Direchlet BC"))


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
            bc = [DirichletBC(V, Constant((0, 0, 0)), patient.markers["BASE"][0])]
            return bc
        
        
    else:
       
        if not (params["base_bc"] == "fix_x"):
            logger.warning("Unknown Base BC {}".format(params["base_bc"]))
            logger.warning("Fix base in x direction")
    
        def base_bc(W):
            '''Make Dirichlet boundary conditions where the base is allowed to slide
            in the x = 0 plane.
            '''
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            bc = [DirichletBC(V.sub(0), 0, patient.markers["BASE"][0])]
            return bc
    
        
        # Apply a linear sprint robin type BC to limit motion
        robin_bc = [[Constant(params["base_spring_k"], 
                                   name ="base_spring_constant"),
                     patient.markers["BASE"][0]]]



    # Circumferential, Radial and Longitudinal basis vector
    crl_basis = {}
    for att in ["circumferential", "radial", "longitudinal"]:
        if hasattr(patient, att):
            crl_basis[att] = getattr(patient, att)

    
    
    solver_parameters = {"mesh": patient.mesh,
                         "facet_function": patient.ffun,
                         "facet_normal": FacetNormal(patient.mesh),
                         "crl_basis": crl_basis,
                         "mesh_function": patient.sfun,
                         "markers":patient.markers,
                         "passive_filling_duration": patient.passive_filling_duration,
                         "strain_weights": patient.strain_weights,
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


def make_control(params, patient):

    ##  Contraction parameter
    if params["gamma_space"] == "regional":
        gamma = RegionalParameter(patient.sfun)
    else:
        gamma_family, gamma_degree = params["gamma_space"].split("_")
        gamma_space = FunctionSpace(patient.mesh, gamma_family, int(gamma_degree))

        gamma = Function(gamma_space, name = 'activation parameter')

        

    ##  Material parameters
    
    # Create an object for each single material parameter
    if params["matparams_space"] == "regional":
        paramvec_ = RegionalParameter(patient.sfun)
        
    else:
        
        family, degree = params["matparams_space"].split("_")
        matparams_space = FunctionSpace(patient.mesh, family, int(degree))
        paramvec_ = Function(matparams_space, name = "matparam vector")


    # Number of passive parameters to optimize
    fixed_matparams_keys = ["fix_a", "fix_a_f", "fix_b", "fix_b_f"]
    npassive = sum([ not params["Optimization_parmeteres"][k] \
                     for k in fixed_matparams_keys])

        
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

    
   


    return paramvec, gamma, matparams

    
    
def get_measurements(params, patient):
    """Get the measurement or the synthetic data
    to be used as BC or targets in the optimization

    :param params: Application parameter
    :param patient: class with the patient data
    :returns: The target data
    :rtype: dict

    """

    # Parameters for the targets
    p = params["Optimization_targets"]
    measurements = {}

    
    # Find the start and end of the measurements
    if params["phase"] == PHASES[0]: #Passive inflation
        # We need just the points from the passive phase
        start = 0
        end = patient.passive_filling_duration

        pvals = params["Passive_optimization_weigths"]
        
        

    elif params["phase"] == PHASES[1]: #Scalar contraction
        # We need just the points from the active phase
        start = patient.passive_filling_duration -1
        end = patient.num_points

        pvals = params["Active_optimization_weigths"]
      
    else:
        # We need all the points 
        start = 0
        end = patient.num_points

        pvals = params["Passive_optimization_weigths"]
        
    
    p["volume"] = pvals["volume"] > 0
    p["rv_volume"] = pvals["rv_volume"] > 0 and patient.mesh_type()=="biv"
    p["regional_strain"] = pvals["regional_strain"] > 0
        
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
        if params["unload"]:
            reference_pressure = 0.0
        else:
            reference_pressure = pressure[0] 
        logger.info("LV Pressure offset = {} kPa".format(reference_pressure))

        #Here the issue is that we do not have a stress free reference mesh. 
        #The reference mesh we use is already loaded with a certain
        #amount of pressure, which we remove.
        pressure = np.subtract(pressure,reference_pressure)
        
        measurements["pressure"] = pressure[start:end]

        if patient.mesh_type() == "biv":
            rv_pressure = np.array(patient.RVP)
            if params["unload"]:
                reference_pressure = 0.0
            else:
                reference_pressure = rv_pressure[0]
            logger.info("RV Pressure offset = {} kPa".format(reference_pressure))
            
            rv_pressure = np.subtract(rv_pressure, reference_pressure)
            measurements["rv_pressure"] = rv_pressure[start:end]
            
        
        
        ## Volume
        if p["volume"]:
            # Calculate difference bwtween calculated volume, and volume given from echo
            volume_offset = get_volume_offset(patient, params)
            logger.info("LV Volume offset = {} cm3".format(volume_offset))

            logger.info("Measured LV volume = {}".format(patient.volume[0]))
            
            
            # Subtract this offset from the volume data
            volume = np.subtract(patient.volume,volume_offset)

            logger.info("Computed LV volume = {}".format(volume[0]))

            measurements["volume"] = volume[start:end]


        if p["rv_volume"]:
            # Calculate difference bwtween calculated volume, and volume given from echo
            volume_offset = get_volume_offset(patient, params, "rv")
            logger.info("RV Volume offset = {} cm3".format(volume_offset))

            logger.info("Measured RV volume = {}".format(patient.RVV[0]))
            
            # Subtract this offset from the volume data
            volume = np.subtract(patient.RVV ,volume_offset)

            logger.info("Computed RV volume = {}".format(volume[0]))
            
            measurements["rv_volume"] = volume[start:end]
                

        if p["regional_strain"]:

            strain = {}
            if hasattr(patient, "strain"):
                for region in patient.strain.keys():
                    strain[region] = patient.strain[region][start:end]
            else:
                msg = ("\nPatient do not have strain as attribute."+
                       "\nStrain will not be used")
                p["regional_strain"] = False
                logger.warning(msg)
            measurements["regional_strain"] = strain
    

    return measurements

def get_volume_offset(patient, params, chamber = "lv"):
    N = FacetNormal(patient.mesh)

    if chamber == "lv":
    
        if patient.mesh_type() == "biv":
            endo_marker = patient.markers["ENDO_LV"][0]
        else:
            endo_marker = patient.markers["ENDO"][0]

        volume = patient.volume[0]
        
    else:
        endo_marker = patient.markers["ENDO_RV"][0]
        volume = patient.RVV[0]

    if volume == -1:
        return 0

    logger.info("Measured = {}".format(volume))
    ds = Measure("exterior_facet",
                 subdomain_data = patient.ffun,
                 domain = patient.mesh)(endo_marker)
    
    X = SpatialCoordinate(patient.mesh)

    if params["unload"] and params["phase"] == PHASES[1]:
        # The cavicty volume is the volume of the uloaded geometry
        # The first measured volume is the unloaded geometry,
        # loaded with the first pressure. Use this to estimate the offset
        family, degree = params["state_space"].split(":")[0].split("_")
        u = Function(VectorFunctionSpace(patient.mesh, family, int(degree)))
        with HDF5File(mpi_comm_world(), params["sim_file"], 'r') as h5file:
            # Get previous state
            group = "/".join([params["h5group"],
                              PASSIVE_INFLATION_GROUP,
                              "displacement","1"])
            h5file.read(u, group)

        # We would like to use interpolate here, but project works with dolfin-adjoint
        u_int = project(u, VectorFunctionSpace(patient.mesh, "CG", 1))
        vol = assemble((-1.0/3.0)*dot(X+u_int,N)*ds)
        logger.info("Computed = {}".format(vol))
    else:
        
        vol = assemble((-1.0/3.0)*dot(X,N)*ds)
    
    return volume - vol

def setup_simulation(params, patient):

    check_patient_attributes(patient)
    # Load measurements
    measurements = get_measurements(params, patient)
    solver_parameters, pressure, controls = make_solver_params(params, patient, measurements)
   
    return measurements, solver_parameters, pressure, controls


class MyReducedFunctional(ReducedFunctional):
    """
    A modified reduced functional of the `dolfin_adjoint.ReducedFuctionl`

    Parameters
    ----------
    for_run: callable
        The forward model, which can be called with the control parameter
        as first argument, and a boolean as second, indicating that annotation is on/off.
    paramvec: :py:class`dolfin_adjoint.function`
        The control parameter
    scale: float
        Scale factor for the functional
    relax: float
        Scale factor for the derivative. Note the total scale factor for the 
        derivative will be scale*relax


    """
    def __init__(self, for_run, paramvec, scale = 1.0, relax = 1.0, verbose = False):

        self.log_level = logger.level
        self.reset()
        self.for_run = for_run
        self.paramvec = paramvec
        
        self.initial_paramvec = gather_broadcast(paramvec.vector().array())
        self.scale = scale
        self.derivative_scale = relax

        self.verbose = verbose
        from optimal_control import has_scipy016
        self.my_print_line = logger.debug if has_scipy016 else logger.info
        
    def __call__(self, value, return_fail = False):


        logger.debug("\nEvaluate functional...")
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
        parameters["adjoint"]["stop_annotating"] = False

       
        # Change loglevel to avoid to much printing (do not change if in dbug mode)
        change_log_level = (self.log_level == logging.INFO) and not self.verbose
        
        if change_log_level:
            logger.setLevel(WARNING)
            
            
        t = Timer("Forward run")
        t.start()

        logger.debug("\nEvaluate forward model")
        self.for_res, crash= self.for_run(paramvec_new, True)
        for_time = t.stop()
        logger.debug(("Evaluating forward model done. "+\
                      "Time to evaluate = {} seconds".format(for_time)))
        self.forward_times.append(for_time)

        if change_log_level:
            logger.setLevel(self.log_level)

        if self.first_call:
            # Store initial results
            self.ini_for_res = self.for_res
            self.first_call = False

	    # Some printing
            self.my_print_line(print_head(self.for_res))
            
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

        grad_norm = None if len(self.grad_norm_scaled) == 0 \
                    else self.grad_norm_scaled[-1]


        self.my_print_line(Text.yellow(print_line(self.for_res, self.iter,
                                            grad_norm, func_value)))
        self.func_values_lst.append(func_value*self.scale)
        self.controls_lst.append(Vector(paramvec_new.vector()))

        
        logger.debug(Text.yellow("Stop annotating"))
        parameters["adjoint"]["stop_annotating"] = True


        if return_fail:
            return self.scale*func_value, crash

        
        return self.scale*func_value

    def reset(self):

        logger.setLevel(self.log_level)
        if not hasattr(self, "ini_for_res"):
            
            self.first_call = True
            self.nr_crashes = 0
            self.iter = 0
            self.nr_der_calls = 0
            self.func_values_lst = []
            self.controls_lst = []
            self.forward_times = []
            self.backward_times = []
            self.grad_norm = []
            self.grad_norm_scaled = []
        else:
            if len(self.func_values_lst): self.func_values_lst.pop()
            if len(self.controls_lst): self.controls_lst.pop()
            if len(self.grad_norm): self.grad_norm.pop()
            if len(self.grad_norm_scaled): self.grad_norm_scaled.pop()


        
    def derivative(self, *args, **kwargs):

        logger.debug("\nEvaluate gradient...")
        self.nr_der_calls += 1
        import math

        t = Timer("Backward run")
        t.start()
        
        out = ReducedFunctional.derivative(self, forget = False)
        back_time = t.stop()
        logger.debug(("Evaluating gradient done. "+\
                      "Time to evaluate = {} seconds".format(back_time)))
        self.backward_times.append(back_time)
        
        for num in out[0].vector().array():
            if math.isnan(num):
                raise Exception("NaN in adjoint gradient calculation.")

        # Multiply with some small number to that we take smaller steps
        gathered_out = gather_broadcast(out[0].vector().array())

        self.grad_norm.append(np.linalg.norm(gathered_out))
        self.grad_norm_scaled.append(np.linalg.norm(gathered_out)*self.scale*self.derivative_scale)
        logger.debug("|dJ|(actual) = {}\t|dJ|(scaled) = {}".format(self.grad_norm[-1],
                                                                   self.grad_norm_scaled[-1]))
        return self.scale*gathered_out*self.derivative_scale


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
