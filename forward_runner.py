from heart_problem import PassiveHeartProblem, ActiveHeartProblem
from setup_optimization import RealValueProjector
from dolfin import *
from dolfin_adjoint import *
from adjoint_contraction_args import *
import numpy as np
from numpy_mpi import *
from utils import Text, list_sum, Object


class BasicForwardRunner(object):
    """
    Runs a simulation using a HeartProblem object
    and compares simulated observations to target data.
    """
    OUTPUT_STR = "{:.5f}\t\t{:.5f}\t  {:.5f}    {:.5f}   {:.5f}  {:.5f}"
    def __init__(self, solver_parameters,
                 p_lv,
                 target_data,
                 endo_lv_marker,
                 crl_basis,
                 params, 
                 spaces):
        
        self.solver_parameters = solver_parameters
        self.p_lv = p_lv 

        self.alpha = params["alpha"]
        self.use_deintegrated_strains = params["use_deintegrated_strains"]

        #Circumferential, radial, longtitudal basis.
        self.crl_basis = crl_basis
        self.endo_lv_marker = endo_lv_marker

        self._set_target_data(target_data)
        self._init_functions(spaces)


    def _set_target_data(self, target_data):

        self.target_strains = target_data.target_strains
        self.target_vols = target_data.target_vols
        self.pressures = target_data.target_pressure

    def _init_functions(self, spaces):

        self.spaces = spaces

        self.V_sim = Function(spaces.r_space, name = "Simulated Volume")
        self.V_diff = Function(spaces.r_space, name = "Volume Difference")
        self.V_meas = Function(spaces.r_space, name = "Target Volume")

        
        self.strain_weights = [Function(spaces.strain_weight_space, \
                                        name = "Strains Weights_{}".format(i)) for i in STRAIN_REGION_NUMS]
        self.strain_weights_deintegrated = self.solver_parameters["strain_weights_deintegrated"]
        

        if self.use_deintegrated_strains:
            self.u_tar = Function(spaces.strainfieldspace, name = "Target Strains")
            self.strain_diffs = Function(spaces.r_space, name = "Strain_Difference")
            
        else:
            self.u_tar = [Function(spaces.strainspace, name = "Target Strains_{}".format(i)) for i in STRAIN_REGION_NUMS]
            self.strain_diffs = [Function(spaces.r_space, name = "Strain_Difference_{}".format(i)) for i in STRAIN_REGION_NUMS]
           
            for i in STRAIN_REGION_NUMS:
                strain_weight = np.zeros(9)
                strain_weight[0] = self.solver_parameters["strain_weights"][i-1][0]
                strain_weight[4] = self.solver_parameters["strain_weights"][i-1][1]
                strain_weight[8] = self.solver_parameters["strain_weights"][i-1][2]
                assign_to_vector(self.strain_weights[i-1].vector(), strain_weight)

            
            
        v = TestFunction(spaces.r_space)
        
        # Volume of the myocardium
        meshvol = gather_broadcast(assemble(Constant(1.0)*v*dx).array())[0]
        self.mesh_vol = Constant(meshvol)

        self.projector = RealValueProjector(TrialFunction(spaces.r_space), v,
                                            self.mesh_vol)
        
        
       
    
    def _save_state(self, state, lv_pressure, volume, strain, strainfield):
        self.states.append(state)
        self.lv_pressures.append(lv_pressure)
        self.volumes.append(volume)

        self.strainfields.append(Vector(strainfield.vector()))

        for region in STRAIN_REGION_NUMS:
            self.strains[region-1].append(Vector(strain[region-1].vector()))
        

    
    def _get_exprval(self, expr, mesh):
        return float(interpolate(expr, FunctionSpace(mesh, "R", 0)))
	
		    
    
    def solve_the_forward_problem(self, annotate = False, phm=None, phase = "passive"):
	
        # Start the clock
        adj_start_timestep(0.0)
        
        #Save Information for later storage.
        self.states = []
        self.volumes = []
        self.lv_pressures = []
        self.strains = [[] for i in STRAIN_REGION_NUMS]
        self.strainfields = []
	
        dx = phm.dx
        
        if self.use_deintegrated_strains:
            strain_matches = (dot(self.strain_weights_deintegrated,phm.strainfield - self.u_tar))**2
            get_strain_error = lambda : assemble(strain_matches*dx)
        else:
            strain_matches = [(dot(self.strain_weights[i-1],phm.strains[i - 1] - self.u_tar[i - 1]))**2 for i in STRAIN_REGION_NUMS]
            get_strain_error = lambda : sum([assemble(sm*dx(i)) for i,sm in zip(STRAIN_REGION_NUMS, strain_matches)])

        functional_values_strain = []
        functional_values_volume = []
        functionals_time = []
        
        if phase == "passive":
            # We must record the initial state
            if self.use_deintegrated_strains:
                self.u_tar.assign(self.target_strains[0], annotate = annotate)
            else:
                for target, strain in zip(self.u_tar, self.target_strains[0]):
                    target.assign(strain, annotate = annotate)

            # And we save it for later reference
            self._save_state(Vector(phm.solver.w.vector()), self._get_exprval(phm.p_lv, phm.mesh),
                             float(phm.get_inner_cavity_volume()), phm.strains, phm.strainfield)


        logger.debug("Volume - Strain interpolation {}".format(self.alpha))
        logger.info("\nLVP (cPa)  LV Volume(ml)  Target Volume(ml)  I_strain  I_volume  I_reg")
	
        strain_diff = list_sum(self.strain_diffs)
	    
        count = 1.0
        
        functional = self.alpha*self.V_diff/self.mesh_vol*dx + (1 - self.alpha)*strain_diff/self.mesh_vol*dx
        

        if phase == "active":
            # Add regulatization term to the functional
            m = phm.solver.parameters['material']['gamma']
            if m.ufl_element().family == "Real":
                reg_term = 0.0
            else:
                reg_term = assemble(self.reg_par*inner(grad(m), grad(m))*dx)
                functional += self.reg_par*inner(grad(m), grad(m))/self.mesh_vol*dx
            

        else:
            # Add the initial state to the recording
            functionals_time.append(functional*dt[0.0])
            reg_term = 0.0

        for strains_at_pressure, target_vol in zip(self.target_strains[1:], self.target_vols[1:]):

            sol, model_strain = phm.next()
	    
            # Assign the target strain 
            if self.use_deintegrated_strains:
                self.u_tar.assign(strains_at_pressure, annotate = annotate)
            else:
                for target, strain in zip(self.u_tar, strains_at_pressure):
                    target.assign(strain, annotate = annotate)

            # Assign the target volume
            self.V_meas.assign(target_vol, annotate = annotate)

            # Compute the strain misfit
            strain_error = get_strain_error()

            #Volume Projections to get dolfin-adjoint to record.            
            self.projector.project(phm.vol, phm.ds, self.V_sim)
            self.projector.project(((self.V_sim - self.V_meas)/self.V_meas)**2, dx, self.V_diff)
            
            #Strain projections to get dolfin-adjoint to record.
            if self.use_deintegrated_strains:
                self.projector.project(strain_matches, dx, self.strain_diffs)
            else:
                for i,sm in zip(STRAIN_REGION_NUMS, strain_matches):
                    self.projector.project(sm, dx(i), self.strain_diffs[i - 1])
                
            
            # Gathering the vector if running in parallell
            v_diff = gather_broadcast(self.V_diff.vector().array())[0]

            self.print_solve_line(phm, strain_error, v_diff, reg_term)

            if phase == "active":
                # There is only on step, so we are done
                adj_inc_timestep(1, True)
                functionals_time.append(functional*dt[1])
            else:
                # Check if we are done with the passive phase
                adj_inc_timestep(count, near(count, len(self.target_vols)-1))
                count += 1
                functionals_time.append(functional*dt[count])
                
                

            functional_values_strain.append(strain_error)
            functional_values_volume.append(v_diff)

            # Save the state
            self._save_state(Vector(phm.solver.w.vector()), self._get_exprval(phm.p_lv, phm.mesh),
                             float(phm.get_inner_cavity_volume()), phm.strains, phm.strainfield)

     
        forward_result = self._make_forward_result(functional_values_strain, functional_values_volume, functionals_time, phm)

        if phase == "active":
            gradient_size = assemble(inner(grad(m), grad(m))*dx)
            forward_result.gamma_gradient = gradient_size
            forward_result.reg_par = float(self.reg_par)
        else:
            forward_result.gamma_gradient = 0.0
            forward_result.reg_par = 0.0

        self.print_finished_report(forward_result)
        return forward_result
    
    def print_finished_report(self, forward_result):
        logger.info("\n\t\tI_strain \tI_volume \tI_reg")
        logger.info("Normal   \t{:.5f}\t\t{:.5f}\t\t{:.5f}".format(forward_result.func_value_strain,forward_result.func_value_volume, forward_result.gamma_gradient))
        logger.info("Weighted \t{:.5f}\t\t{:.5f}\t\t{:.5f}".format(forward_result.weighted_func_value_strain,forward_result.weighted_func_value_volume, forward_result.gamma_gradient*forward_result.reg_par))
    
    def print_solve_line(self, phm, strain_error, v_diff, reg_term):
        v_sim = gather_broadcast(self.V_sim.vector().array())[0]
        v_meas = gather_broadcast(self.V_meas.vector().array())[0]

        # self.logger.debug(self.OUTPUT_STR.format(self._get_exprval(phm.p_lv, phm.mesh),
		# 			 self._get_exprval(phm.p_rv, phm.mesh),
		# 			 v_sim, v_meas, strain_error, v_diff))

        logger.info(self.OUTPUT_STR.format(self._get_exprval(phm.p_lv, phm.mesh),
					 v_sim, v_meas, strain_error, v_diff, reg_term))

    def _make_forward_result(self, functional_values_strain, functional_values_volume, functionals_time, phm):
        fr = Object()
        fr.phm = phm

        fr.total_functional = list_sum(functionals_time)

        fr.func_value_strain = sum(functional_values_strain)
        fr.func_value_volume = sum(functional_values_volume)

        fr.weighted_func_value_strain = (1 - self.alpha)*fr.func_value_strain
        fr.weighted_func_value_volume = self.alpha*fr.func_value_volume

        fr.func_value = fr.weighted_func_value_strain + fr.weighted_func_value_volume
        fr.states = self.states
        fr.volumes = self.volumes
        fr.lv_pressures = self.lv_pressures
        fr.strains = self.strains
        fr.strainfields = self.strainfields

        fr.strain_weights = self.solver_parameters["strain_weights"]
        fr.strain_weights_deintegrated = self.solver_parameters["strain_weights_deintegrated"]
        
        
        return fr


class ActiveForwardRunner(BasicForwardRunner):
    def __init__(self, 
                 solver_parameters, 
                 p_lv, 
                 target_data,
                 params,
                 gamma_previous,
                 patient,
                 spaces):



        # Store file with information about passive phase
        self.h5filepath = params["sim_file"]
        self.active_contraction_iteration_number = params["active_contraction_iteration_number"]
        self.gamma_previous = gamma_previous
        self.reg_par = Constant(params["reg_par"])

        self.passive_filling_duration = patient.passive_filling_duration
        self.strain_markers = patient.strain_markers
        self.endo_lv_marker = patient.ENDO
        self.crl_basis = (patient.e_circ, patient.e_rad, patient.e_long)


        
        BasicForwardRunner.__init__(self, solver_parameters, p_lv, 
                               target_data, self.endo_lv_marker,
                               self.crl_basis, params, spaces)

        
        

        self.solver_parameters['material']['gamma'].assign(gamma_previous, annotate = True)

        self.cphm = ActiveHeartProblem(self.pressures,
                                             self.solver_parameters,
                                             self.p_lv,
                                             self.endo_lv_marker,
                                             self.crl_basis,
                                             spaces,
                                             self.passive_filling_duration, 
                                             params,
                                             annotate = False)
	
        logger.debug("\nVolume before pressure change: {:.3f}".format(self.cphm.get_inner_cavity_volume()))
        self.cphm.increase_pressure()
        logger.debug("Volume after pressure change: {:.3f}".format(self.cphm.get_inner_cavity_volume()))

    def __call__(self, m,  annotate = False):
	    
        
        logger.info("\nEvaluating Model")
        

        # Take small steps with gamma until we have only one point left
        # We do not want to record this as we only want to optimize the final value
        logger.debug(Text.yellow("Stop annotating"))
        parameters["adjoint"]["stop_annotating"] = True
        try:
            # Try to step up gamma to the given one
            logger.debug("Try to step up gamma")
            w_old = self.cphm.get_state()
            gamma_old = self.gamma_previous.copy(True)
            self.cphm.next_active(m, self.gamma_previous)
	    
        except StopIteration:
            logger.debug("Stepping up gamma failed")
            # If stepping up gamma fails, assign the previous gamma
            # and return a crash=True, so that the Reduced functional
            # knows that we were unable to step up gamma
            logger.debug(Text.yellow("Start annotating"))
            parameters["adjoint"]["stop_annotating"] = not annotate

            logger.debug("Assign the old state and old gamma")
            # Assign the old state
            self.cphm.solver.w.assign(w_old, annotate=annotate)
            # Assign the old gamma
            m.assign(gamma_old)
            self.cphm.solver.parameters['material']['gamma'].assign(gamma_old)
            self.solver_parameters['material']['gamma'].assign(gamma_old)
            self.gamma_previous.assign(gamma_old)

            # Solve the forward problem with the old gamma
            logger.debug("Solve the forward problem with the old gamma")
            forward_result = BasicForwardRunner.solve_the_forward_problem(self, annotate, self.cphm, "active")

            return forward_result, True

        else:
            # Stepping up gamma succeded
            logger.debug("Stepping up gamma succeded")
            # Get the current state
            w = self.cphm.get_state()
            logger.debug(Text.yellow("Start annotating"))
            parameters["adjoint"]["stop_annotating"] = not annotate

            # Assign the state where we have only one step with gamma left, and make sure
            # that dolfin adjoint record this.
            logger.debug("Assign the new state and gamma")
            self.cphm.solver.w.assign(w, annotate=annotate)

            # Now we make the final solve
            self.cphm.solver.parameters['material']['gamma'].assign(m)
            self.solver_parameters['material']['gamma'].assign(m)
            self.gamma_previous.assign(m)

            logger.debug("Solve the forward problem with the new gamma")
            forward_result = BasicForwardRunner.solve_the_forward_problem(self, annotate, self.cphm, "active")

            return forward_result, False




class PassiveForwardRunner(BasicForwardRunner):    
    def __call__(self, m, annotate = False, phm=None):
        m = split(m)
        self.solver_parameters["material"]["a"] = m[0]
        self.solver_parameters["material"]["b"] = m[1]
        self.solver_parameters["material"]["a_f"] = m[2]
        self.solver_parameters["material"]["b_f"] = m[3]
       
     
        phm = PassiveHeartProblem(self.pressures, 
                                  self.solver_parameters,
                                  self.p_lv, self.endo_lv_marker, 
                                  self.crl_basis, self.spaces)

    
        forward_result = BasicForwardRunner.solve_the_forward_problem(self, annotate, phm, "passive")


        return forward_result
