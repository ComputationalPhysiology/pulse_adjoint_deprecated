"""
This file contains the module for solving the 
forward problem. It also records the forwards solve, 
so that dolfin-adjoint can run the backward solve.
"""
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
from heart_problem import PassiveHeartProblem, ActiveHeartProblem
from dolfinimport import *
from optimization_targets import *
from adjoint_contraction_args import *
import numpy as np
from numpy_mpi import *
from utils import Text, list_sum, Object, TablePrint, UnableToChangePressureExeption
from setup_optimization import RegionalParameter
from lvsolver import SolverDidNotConverge



class BasicForwardRunner(object):
    """
    Runs a simulation using a HeartProblem object
    and compares simulated observations to target data.

    :param dict solver_parameters: solver parameters coming from 
                                   setup_optimization.make_solver_paramerters()
    :param list pressure: list of pressure that should be solved for, 
                          starting with the current pressure
    :param dict bcs: Dictionary with boundary conditions coming from
                     run_optimization.load_target_data()
    :param dict optimization_targets: Dictionary with optimization targets,  
                                      coming from run_optimization.load_target_data()
    :param dict params: adjoint contraction paramters
    """    
    
    def __init__(self,
                 solver_parameters,
                 pressure, 
                 bcs,
                 optimization_targets,
                 params):
        """Initialize base class for forward solver

        """
        

        
        self.bcs = bcs
        self.mesh_type = params["Patient_parameters"]["mesh_type"]
        self.solver_parameters = solver_parameters

        self.pressure = pressure
            
        self.target_params = params["Optimization_targets"]
        self.params = params

        self.meshvol = Constant(assemble(Constant(1.0)*dx(solver_parameters["mesh"])),
                                name = "mesh volume")

        
        # Initialize target functions
        for target in optimization_targets.values():
            target.set_target_functions()

        self.regularization = optimization_targets.pop("regularization", None)
        
        self.optimization_targets = optimization_targets

    def _print_head(self):
        """
        Print the top line for the output of the forward solve
        """
        
        head = "{:<12}".format("LV Pressure")
        if self.mesh_type == "biv":
            head += "{:<12}".format("RV Pressure") 
        for key,val in self.target_params.iteritems():
                if val: head+= self.optimization_targets[key].print_head()

        head += self.regularization.print_head()
        return head

    def _print_line(self, it):
        """
        Print each line for the forward solve, corresponding to the head
        """
        
        line= "{:<12.2f}".format(self.bcs["pressure"][it])
        if self.mesh_type == "biv":
            line += "{:<12.2f}".format(self.bcs["rv_pressure"][it]) 
        for key,val in self.target_params.iteritems():
            if val: line+= self.optimization_targets[key].print_line()

        line += self.regularization.print_line()

        return line

    def _print_functional(self):
        """
        Print the terms in the functional in a mathematical way
        """

        s = ("\n"+ "Functional".center(100, "-") +\
             "\n"+"J = {}".format((len(self.opt_weights.keys())*" {{}}*I_{} +").\
                                 format(*self.opt_weights.keys())[:-1].\
                                  format(*self.opt_weights.values())).center(100, " ")+ "\n")
        return s
             
        

    def solve_the_forward_problem(self, phm, annotate = False, phase = "passive"):
        """Solve the forward model

        :param bool annotate: 
        :param phm: A heart problem instance
        :param phase: Which phase of the cycle, options: ['passive', 'active']
        :returns: A dictionary with the results
        :rtype: dict

        """
   
        # Set the functional value for each target to zero
        for key,val in self.target_params.iteritems():
            if val: self.optimization_targets[key].reset()
        self.regularization.reset()
        
        # Start the clock
        adj_start_timestep(0.0)

      
        #Save Information for later storage.
        self.states = []
        
        functional_values = []
        functionals_time = []

        if phase == "passive":
            for key,val in self.target_params.iteritems():
                if val: self.optimization_targets[key].next_target(0, annotate=annotate)
            
            # And we save it for later reference
            phm.solver.solve()
            self.states.append(Vector(phm.solver.get_state().vector()))


        # Print the functional
        logger.info(self._print_functional())                               
        # Print the head of table 
        logger.info(self._print_head())
       
        functional = self.make_functional()
              
       
        if phase == "active":
            # Add regulatization term to the functional
            m = phm.solver.parameters['material'].gamma
            
        else:

            # FIXME : assume for now that only a is optimized
            m = self.paramvec
        
            # Add the initial state to the recording
            functionals_time.append(functional*dt[0.0])
            
               
        for it, p in enumerate(self.bcs["pressure"][1:], start=1):
        
            sol = phm.next()

        
            if self.params["passive_weights"] == "all" \
               or (self.params["passive_weights"] == "all" \
                   and it == len(self.bcs["pressure"])-1) \
                   or int(self.params["passive_weights"]) == it:


                self.update_targets(it, split(sol)[0], m, annotate = annotate)
                
            
                # Print the values
                logger.info(self._print_line(it))
            

                if phase == "active":
                    # There is only on step, so we are done
                    adj_inc_timestep(1, True)
                    functionals_time.append(functional*dt[1])
                else:
                    # Check if we are done with the passive phase
                    
                    adj_inc_timestep(it, it == len(self.bcs["pressure"])-1)
                    functionals_time.append(functional*dt[it+1])
     
                functional_values.append(assemble(functional))
                
            self.states.append(Vector(phm.solver.get_state().vector()))
            
        forward_result = self._make_forward_result(functional_values,
                                                   functionals_time)

        # self._print_finished_report(forward_result)
        return forward_result

    def make_functional(self):

        # Get the functional value of each term in the functional
        func_lst = []
        for key,val in self.target_params.iteritems():
            if val:
                func_lst.append(self.opt_weights[key]*self.optimization_targets[key].get_functional())

        # Collect the terms in the functional
        functional = list_sum(func_lst)
        # Add the regularization term
        functional += self.regularization.get_functional()

        return functional
    
    def update_targets(self, it, u, m, annotate=False):

        for key,val in self.target_params.iteritems():
            
            if val:
                       
                self.optimization_targets[key].next_target(it, annotate=annotate)
                self.optimization_targets[key].assign_simulated(u)
                self.optimization_targets[key].assign_functional()
                self.optimization_targets[key].save()

                        
        self.regularization.assign(m, annotate = annotate)
        self.regularization.save()
        
    
    def _print_finished_report(self, forward_result):


        targets = forward_result["optimization_targets"]
        reg  = forward_result["regularization"]

        keys = targets.keys()+["regularization"]
        values = [sum(t.results["func_value"]) for t in targets.values()] + \
                 [sum(reg.results["func_value"])]
        
        n = len(keys)
        
        logger.info("\nMismatch functional values:")
        logger.info("\t"+(n*"{:10}\t").format(*keys))
        logger.info("\t"+(n*"{:10.4e}\t").format(*values))


    def _make_forward_result(self, functional_values, functionals_time):

        target_values = {}
        for k, v in self.optimization_targets.iteritems():
            target_values[k] = v.func_value
            
        target_values["regularization"] = self.regularization.func_value
        
        
        fr = {"optimization_targets": self.optimization_targets,
              "target_values": target_values,
              "regularization": self.regularization,
              "states": self.states,
              "bcs": self.bcs,
              "total_functional": list_sum(functionals_time),
              "func_value": sum(functional_values)}

        return fr


class ActiveForwardRunner(BasicForwardRunner):
    """
    The active forward runner 

    Parameters
    ----------
        
    solver_parameters : dict
        solver parameters coming from setup_optimization.make_solver_paramerters()
    pressure : list 
        list of pressure that should be solved for, starting with the current pressure
    bcs : dict
        Dictionary with boundary conditions coming from run_optimization.load_target_data()
    optimization_targets : dict
        Dictionary with optimization targets, coming from run_optimization.load_target_data()
    params : dict
        adjoint contraction paramters
    gamma_previous: :py:class`dolfin.function`
        The active contraction parameter


    **Example of usage**::

          # Setup compiler parameters
          setup_general_parameters()
          params = setup_adjoint_contraction_parameter()
          params['phase'] = 'active_contraction'
          # Initialize patient data
          patient = initialize_patient_data(param['Patient_parameters'], False)

          # Start with the first point with active contraction.
          # Make sure to run the passive inflation first!
          params["active_contraction_iteration_number"] = 0

          #Load patient data, and set up the simulation
          measurements, solver_parameters, pressure, gamma = setup_simulation(params, patient)

          # Load targets
          optimization_targets, bcs = load_targets(params, solver_parameters, measurements)

       
          #Initialize the solver for the Forward problem
          for_run = ActiveForwardRunner(solver_parameters, 
                                        pressure, 
                                        bcs,
                                        optimization_targets,
                                        params, 
                                        paramvec)
    """
    def __init__(self, 
                 solver_parameters, 
                 pressure, 
                 bcs,
                 optimization_targets,
                 params,
                 gamma_previous):
        



        # Store file with information about passive phase
        self.h5filepath = params["sim_file"]
        self.outdir = params["outdir"]
        self.active_contraction_iteration_number = params["active_contraction_iteration_number"]
        self.gamma_previous = gamma_previous
        
        
        
        BasicForwardRunner.__init__(self,
                                    solver_parameters,
                                    pressure,
                                    bcs,
                                    optimization_targets,
                                    params)

        self.opt_weights = {}
        for k, v in params["Active_optimization_weigths"].iteritems():
            if k in self.optimization_targets.keys() or \
               k == "regularization":
                self.opt_weights[k] = v
        
        

        self.solver_parameters['material'].gamma.assign(gamma_previous, annotate = True)

        self.cphm = ActiveHeartProblem(self.bcs,
                                       self.solver_parameters,
                                       self.pressure,
                                       self.params,
                                       annotate = False)

        self.cphm.increase_pressure()

        
        
    def __call__(self, m,  annotate = False):
	    
        logger.info("Evaluating model")
        # Take small steps with gamma until we have only one point left
        # We do not want to record this as we only want to optimize the final value
        logger.debug(Text.yellow("Stop annotating"))
        parameters["adjoint"]["stop_annotating"] = True
   
        logger.debug("Try to step up gamma")
        
        w_old = self.cphm.get_state()
        gamma_old = self.gamma_previous.copy()
        
        try:
            self.cphm.next_active(m, self.gamma_previous.copy())
    	    
        except SolverDidNotConverge as ex:
            logger.debug("Stepping up gamma failed")

            # Save the gamma
            file_format = "a" if os.path.isfile(self.outdir+"/gamma_crash.h5") else "w"

            p = 0
            acin = self.active_contraction_iteration_number
            if file_format == "a":
                import h5py
                h5pyfile = h5py.File(self.outdir+"/gamma_crash.h5", "r")
                if "point_{}".format(acin) in h5pyfile.keys():
                    while "crash_point_{}".format(p) in h5pyfile["point_{}".format(acin)].keys():
                        p += 1

                with HDF5File(mpi_comm_world(), self.outdir+"/gamma_crash.h5", file_format) as h5file:
                    h5file.write(m, "point_{}/crash_point_{}".format(acin, p))
                


            logger.debug("Assign the old state and old gamma")
            # Assign the old state
            self.cphm.solver.get_state().assign(w_old)
            # Assign the old gamma
            self.cphm.solver.parameters['material'].gamma.assign(gamma_old)
            self.gamma_previous.assign(gamma_old)

           
            raise ex



        else:
            # Stepping up gamma succeded
            logger.debug("Stepping up gamma succeded")
            # Get the current state
            w = self.cphm.get_state()
          
            self.gamma_previous.assign(m)
            logger.debug(Text.yellow("Start annotating"))
            parameters["adjoint"]["stop_annotating"] = not annotate
            
            # Assign the state where we have only one step with gamma left, and make sure
            # that dolfin adjoint record this.
            logger.debug("Assign the new state and gamma")
            self.cphm.solver.get_state().assign(w, annotate=annotate)
            
            # Now we make the final solve
            self.cphm.solver.parameters['material'].gamma.assign(m)
                        
            w = self.cphm.get_state()
                      
            logger.debug("Solve the forward problem with the new gamma")
            
            forward_result = BasicForwardRunner.solve_the_forward_problem(self, self.cphm,
                                                                          annotate, "active")
            w = self.cphm.get_state()
           
            
            return forward_result, False




class PassiveForwardRunner(BasicForwardRunner):
    """
    The passive forward runner

    **Example of usage**::

          # Setup compiler parameters
          setup_general_parameters()
          params = setup_adjoint_contraction_parameter()
          params['phase'] = 'passive_inflation'
          # Initialize patient data
          patient = initialize_patient_data(param['Patient_parameters'], False)

          #Load patient data, and set up the simulation
          measurements, solver_parameters, pressure, paramvec = setup_simulation(params, patient)

          # Load targets
          optimization_targets, bcs = load_targets(params, solver_parameters, measurements)

       
          #Initialize the solver for the Forward problem
          for_run = PassiveForwardRunner(solver_parameters, 
                                         pressure, 
                                         bcs,
                                         optimization_targets,
                                         params, 
                                         paramvec)
    """
    def __init__(self, solver_parameters, pressure, 
                 bcs, optimization_targets,
                 params, paramvec):
        """
        Initialize class for passive forward solver

        :param dict solver_parameters: solver parameters coming from 
                                       setup_optimization.make_solver_paramerters()
        :param list pressure: list of pressure that should be solved for, 
                              starting with the current pressure
        :param dict bcs: Dictionary with boundary conditions coming from
                         run_optimization.load_target_data()
        :param dict optimization_targets: Dictionary with optimization targets,  
                                          coming from run_optimization.load_target_data()
        :param dict params: adjoint contraction paramters
        :param paramvec: A function containing the material parameters
                         which should be optimized
        :type paramvec: :py:class`dolfin.function`

        """
        
        
        BasicForwardRunner.__init__(self,
                                    solver_parameters,
                                    pressure, 
                                    bcs,
                                    optimization_targets,
                                    params)
   
        self.opt_weights = {}
        for k, v in params["Passive_optimization_weigths"].iteritems():
            if k in self.optimization_targets.keys() or \
               k == "regularization":
                self.opt_weights[k] = v
                
        self.paramvec = paramvec
        self.cphm  =  self.get_phm(annotate, return_state =False)

    def __call__(self, m, annotate = False):

        self.assign_material_parameters(m)
        self.cphm  =  self.get_phm(annotate, return_state =False)
        parameters["adjoint"]["stop_annotating"] = not annotate
        forward_result = BasicForwardRunner.solve_the_forward_problem(self, self.cphm, annotate, "passive")
      
        
        return forward_result, False



    def assign_material_parameters(self, m):

        self.paramvec.assign(m)
        npassive = sum([ not self.params["Optimization_parmeteres"][k] \
                     for k in ["fix_a", "fix_a_f", "fix_b", "fix_b_f"]])
    

        lst = ["fix_a", "fix_a_f", "fix_b", "fix_b_f"]

        
        if npassive == 1:
            fixed_idx = np.nonzero([not self.params["Optimization_parmeteres"][k] for k in lst])[0][0]
            par = lst[fixed_idx].split("fix_")[-1]
            if self.params["matparams_space"] == "regional":
                paramvec = project(self.paramvec.get_function(), self.paramvec.get_ind_space())
            else:
                paramvec = self.paramvec
                
            mat = getattr(self.solver_parameters["material"], par)
            mat.assign(paramvec)
        else:
            paramvec_split = split(self.paramvec)
            fixed_idx = np.nonzero([not self.params["Optimization_parmeteres"][k] for k in lst])[0]

            
            for it, idx in enumerate(fixed_idx):
                par = lst[idx].split("fix_")[-1]

                if self.params["matparams_space"] == "regional":
                    rg = RegionalParameter(self.paramvec._meshfunction)
                    rg.assign(project(paramvec_split[it], rg.function_space()))
                    v = rg.get_function()
                else:
                    v = paramvec_split[it]
                    
                mat = getattr(self.solver_parameters["material"], par)
                # mat.assign(v)
                mat = v
        

        

    def get_phm(self, annotate, return_state = False):

        phm = PassiveHeartProblem(self.bcs,
                                  self.solver_parameters,
                                  self.pressure)

        if return_state:
            w_old = phm.get_state().copy(deepcopy=True)

        # Do an initial solve for the initial point
        parameters["adjoint"]["stop_annotating"] = True
        phm.solver.solve()
        parameters["adjoint"]["stop_annotating"] = not annotate

        if return_state:
            return phm, w_old
        
        return phm
