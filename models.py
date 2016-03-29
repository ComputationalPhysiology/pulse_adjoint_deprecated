
from dolfin import *
from dolfin_adjoint import *
from adjoint_contraction_args import *
import math
import numpy as np
from numpy_mpi import *
from utils import Text


class BasicForwardRunner(object):
    """
    Runs a simulation using a PressureHeartModel object
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


class BasicHeartProblem(collections.Iterator):
    """
    Runs a biventricular simulation of the diastolic phase of the cardiac
    cycle. The simulation is driven by LV and RV pressures and is quasi-static.
    """
    def __init__(self, pressure, solverparams, p_lv, endo_lv_marker, 
                 crl_basis, spaces):

        self._init_pressures(pressure, p_lv)
        
        self._init_measures_and_markers(endo_lv_marker, solverparams)

        #Objects needed for Volume calculation
        self._init_strain_functions(spaces)
        
        self.crl_basis = crl_basis

        # Mechanical solver Active strain Holzapfel and Ogden
        self.solver = ActiveHaoSolver(solverparams)
        
        self.p_lv.t = self.pressure[0]
     
	
    
    def next(self):
        p_next = self.pressure_gen.next()
        p_prev = self.p_lv.t
        p_diff = abs(p_next - p_prev)

        converged = False
        nsteps = max(2, int(math.ceil(p_diff/PRESSURE_INC_LIMIT)))
        n_max = 60
        pressures = np.linspace(p_prev, p_next, nsteps)
        
        while not converged and nsteps < n_max:
            try:
                for p in pressures[1:]:
                    
                    self.p_lv.t = p
                    
                    out = self.solver.solve()
                    p_prev = p

                converged = True
            except RuntimeError:
                self.logger.warning(Text.red("Solver chrashed when increasing pressure from {} to {}".format(p_prev, p_next)))
                self.logger.warning("Take smaller steps")
                nsteps += 2
                pressures = np.linspace(p_prev, p_next, nsteps)
                

        if nsteps == n_max:
            raise RuntimeError("Unable to increase pressure")

        strains = self.project_to_strains(self.u)
        
        return out, strains

    
    def get_state(self):
        return self.solver.w.copy(True)

    def project_to_strains(self, u):
        gradu = grad(u)
        grad_u_diag = as_vector([inner(e,gradu*e) for e in self.crl_basis])

        
        solve(inner(self.strainfield_u, self.strainfield_v)*self.dx == inner(self.strainfield_v, grad_u_diag)*self.dx,self.strainfield)
    
        for i in STRAIN_REGION_NUMS:
            solve(inner(self.strain_u, self.strain_v)*self.dx(i) == \
                  inner(self.strain_v, grad_u_diag)*self.dx(i), \
                  self.strains[i - 1])

        return self.strains

    def get_inner_cavity_volume(self):
        return assemble(self.vol_form)

    def _init_pressures(self, pressure, p_lv):
        self.pressure = pressure
        self.pressure_gen = (p for p in pressure[1:])
        self.p_lv = p_lv
        

    def _init_strain_functions(self, spaces):
        
        
        self.strainfieldspace = spaces.strainfieldspace
        self.strainfield = Function(self.strainfieldspace)
        
        self.strainspace = spaces.strainspace
        self.strains = [Function(self.strainspace,
                                     name = "Simulated Strain_{}".format(i)) for i in STRAIN_REGION_NUMS]

        self.strain_u = TrialFunction(self.strainspace)
        self.strain_v = TestFunction(self.strainspace)

        self.strainfield_u = TrialFunction(self.strainfieldspace)
        self.strainfield_v = TestFunction(self.strainfieldspace)
        
       
    def _init_measures_and_markers(self, endo_lv_marker, solverparams):
        # Boundary markers
        ffun = solverparams["facet_function"]
        # Mesh
        self.mesh = solverparams["mesh"]
        # Surface measure
        self.ds = Measure("exterior_facet", subdomain_data = ffun, domain = self.mesh)(endo_lv_marker)
        # Volume measure
        self.dx = Measure("dx", subdomain_data = solverparams["mesh_function"],
                                domain = solverparams["mesh"])
        
        self.strain_markers = solverparams["mesh_function"]
        

    def _init_volume_forms(self):
        # Reference coordinates
        X = SpatialCoordinate(self.mesh)

        # Facet Normal 
        N = self.solver.parameters["facet_normal"]

        # Collect displacement u
        self.u, p = split(self.solver.w)

        # Deformation gradient
        F = grad(self.u) + Identity(3)
        J = det(F)

        # Compute volume
        self.vol = (-1.0/3.0)*dot(X + self.u, J*inv(F).T*N)
        self.vol_form = self.vol*self.ds



def get_mean(f):
    return gather_broadcast(f.vector().array()).mean()


def get_max(f):
    return gather_broadcast(f.vector().array()).max()

def get_max_diff(f1,f2):
    diff = f1.vector() - f2.vector()
    diff.abs()
    return diff.max()





class SyntheticHeartProblem(BasicHeartProblem):
    """
    I already have gamma. Now run a simulation using the list of gamma.
    """
    def __init__(self, pressure, solver_parameters, p_lv, endo_lv_marker, crl_basis, spaces, gamma_list):
        
        self.gamma_gen = (g for g in gamma_list)
        self.gamma = gamma_list[0]
        
        BasicHeartProblem.__init__(self, pressure, solver_parameters, p_lv, endo_lv_marker, crl_basis, spaces)

        BasicHeartProblem._init_volume_forms(self)

    def next(self):

        nr_steps = 2
        g_prev = self.gamma
        gamma_current = self.gamma_gen.next()
        
        
        logger.debug("\tGamma:    Mean    Max")
        logger.debug("\tPrevious  {:.3f}  {:.3f}".format(get_mean(g_prev), 
                                                         get_max(g_prev)))
                                                         
        logger.debug("\tNext      {:.3f}  {:.3f} ".format(get_mean(gamma_current), 
                                                         get_max(gamma_current)))


       

        dg = Function(g_prev.function_space())
        dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] - g_prev.vector()[:])
        g = Function(g_prev.function_space())
        g.assign(g_prev)

        out = self.get_state()

        
        done = False
        
        
        
        
        while not done:
                
            for i in range(1, nr_steps+1):
     
                g.vector()[:] +=  dg.vector()[:]

                self.solver.parameters['material']['gamma'].assign(g)

                try:
                    # mpi_print("try1: mean gamma = {}".format(gather_broadcast(g.vector().array()).mean()))
                    out = self.solver.solve()
                except RuntimeError:
                    logger.warning("Solver crashed. Reduce gamma step")
                    
                    nr_steps += 4
                    g.assign(g_prev)
                    dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] - g_prev.vector()[:]) 
                    mpi_print("DG vector max {}".format(dg.vector().max()))
                    
                    break

                else:
                    g_prev.assign(g)
                    
                        
                if i == nr_steps:
                    done = True

        self.gamma = gamma_current

        return BasicHeartProblem.next(self)



class ActiveBasicHeartProblem(BasicHeartProblem):
    def __init__(self,
                 pressure,
                 solver_parameters,
                 p_lv,
                 endo_lv_marker,
                 crl_basis,
                 spaces,
                 passive_filling_duration, 
                 params,
                 annotate = False):
        '''
        A pressure heart model for the regional contracting gamma.
        '''            
        

        self.alpha = params["alpha"]
        self.passive_filling_duration = passive_filling_duration
       
        

        BasicHeartProblem.__init__(self, pressure, solver_parameters, p_lv, 
                                    endo_lv_marker, crl_basis, spaces)


        w_temp = Function(self.solver.W, name = "w_temp")
        with HDF5File(mpi_comm_world(), params["sim_file"], 'r') as h5file:
        
            # Get previous regional gamma and state
            if params["active_contraction_iteration_number"] == 0:
                h5file.read(w_temp, PASSIVE_INFLATION_GROUP.format(params["alpha_matparams"]) + \
                            "/states/{}".format(passive_filling_duration - 1))
            else:
                h5file.read(w_temp, ACTIVE_CONTRACTION_GROUP.
                            format(params["alpha"],
                                   params["active_contraction_iteration_number"] - 1) + "/states/0")


        self.solver.w.assign(w_temp, annotate = annotate)
        BasicHeartProblem._init_volume_forms(self)
       
    def next(self):
        out = self.solver.solve()
	
        strains = self.project_to_strains(self.u)
        return out, strains

    
    def increase_pressure(self):
        
        
        p_next = self.pressure_gen.next()
        p_prev = self.p_lv.t 
        p_diff = abs(p_next - p_prev)
        
        logger.debug("Increase pressure:  previous   next")
        logger.debug("\t            {:.3f}     {:.3f}".format(p_prev, p_next))

        converged = False
        nsteps = max(2, int(math.ceil(p_diff/PRESSURE_INC_LIMIT)))
        n_max = 100
        pressures = np.linspace(p_prev, p_next, nsteps)
        

        while not converged and nsteps < n_max:
            try:
                for p in pressures[1:]:
                    self.p_lv.t = p
                   
                    out = self.solver.solve()
                    p_prev = p
                converged = True
            except RuntimeError:
                logger.warning("Solver chrashed when increasing pressure from {} to {}".format(p_prev, p))
                logger.warning("Take smaller steps")
                nsteps += 4
                pressures = np.linspace(p_prev, p_next, nsteps)
                

        if nsteps >= n_max:
            raise RuntimeError("Unable to increase pressure")

        
        return out
    
   
    
        
    
    def next_active(self, gamma_current, g_prev):

        max_diff = get_max_diff(gamma_current, g_prev)
        nr_steps = max(2, int(math.ceil(max_diff/GAMMA_INC_LIMIT)))

        logger.debug("\tGamma:    Mean    Max     max difference")
        logger.debug("\tPrevious  {:.3f}  {:.3f}    {:.3e}".format(get_mean(g_prev), 
                                                                   get_max(g_prev), 
                                                                   max_diff))
        logger.debug("\tNext      {:.3f}  {:.3f} ".format(get_mean(gamma_current), 
                                                          get_max(gamma_current)))

        dg = Function(g_prev.function_space(), name = "dg")
        dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] - g_prev.vector()[:])


        g = Function(g_prev.function_space(), name = "g")
        g.assign(g_prev)

        # Store the old stuff
        w_old = self.get_state()
        g_old = g_prev.copy()

        idx = 1
        done = False
        finished_stepping = False
             
        # If the solver crashes n times it is possibly stuck
        MAX_NR_CRASH = 5
        nr_crashes = 0

        logger.debug("\n\tIncrement gamma...")
        logger.debug("\tMean \tMax")
        while not done:
            while not finished_stepping:

                if nr_crashes > MAX_NR_CRASH:
                    self.solver.parameters['material']['gamma'].assign(g_old)
                    self.solver.w.assign(w_old)
                    raise StopIteration("Iteration have chrashed too many times")

                
                # Loop over the steps
                for i in range(1, nr_steps):
                
                    # Increment gamma
                    g.vector()[:] +=  dg.vector()[:]
                    # Assing the new gamma
                    self.solver.parameters['material']['gamma'].assign(g)
                        
                    try:
                        # Try to solve
                        logger.debug("\t{:.3f} \t{:.3f}".format(get_mean(g), get_max(g)))
                        out = self.solver.solve()

                    except RuntimeError as ex:
                        # If that does not work increase the number of steps
                        logger.warning("Solver crashed. Reduce gamma step")
                        nr_steps += 4
                        
                        # Assign the previous gamma
                        g.assign(g_prev)
                        dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] - g_prev.vector()[:]) 
                        nr_crashes += 1

                        break

                    else:
                        g_prev.assign(g)
                        idx += 1
                        
                if i == nr_steps-1:
                    finished_stepping = True

            # All points up to the last point converged. 
            # Now check that the last point also converges.
            
            # Store the current solution
            w = self.get_state()

            self.solver.parameters['material']['gamma'].assign(gamma_current)
            
            try:
                logger.debug("\t{:.3f} \t{:.3f}".format(get_mean(gamma_current), 
                                                       get_max(gamma_current)))
                out = self.solver.solve()
            except RuntimeError:
                nr_steps *= 2
                logger.warning("\tFinal solve-step crashed. Reduce gamma step")
                g.assign(g_prev)
                dg.vector()[:] = 1./(nr_steps) * (gamma_current.vector()[:] - g_prev.vector()[:]) 

                finished_stepping = False
            else:
                # Assign the previous state
                self.solver.w.assign(w, annotate = False)
                self.solver.parameters['material']['gamma'].assign(g_prev)
              
                done = True
                
        


        return out


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





# Passive Models
class PassiveHeartProblem(BasicHeartProblem):
    def __init__(self, pressure, solver_parameters, p_lv, 
                 endo_lv_marker, crl_basis, spaces):
       
        BasicHeartProblem.__init__(self, pressure, solver_parameters, p_lv, 
                                    endo_lv_marker, crl_basis, spaces)

        BasicHeartProblem._init_volume_forms(self)


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

    
        # t = Timer("Forward solve")
        # t.start()
        forward_result = BasicForwardRunner.solve_the_forward_problem(self, annotate, phm, "passive")

        # t.stop()
        # list_timings(TimingClear_keep, [TimingType_wall,])
        # exit()

        return forward_result
