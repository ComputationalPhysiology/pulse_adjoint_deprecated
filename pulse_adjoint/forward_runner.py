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
from utils import Text, list_sum, Object, TablePrint



class BasicForwardRunner(object):
    """
    Runs a simulation using a HeartProblem object
    and compares simulated observations to target data.
    """    
    
    def __init__(self,
                 solver_parameters,
                 pressure, 
                 bcs,
                 optimization_targets,
                 params):
        """Initialize base class for forward solver

        :param solver_parameters: solver parameters coming from 
        setup_optimization.make_solver_paramerters()
        :param pressure: list of pressure that should be solved for, 
        starting with the current pressure
        :param bcs: Dictionary with boundary conditions coming from
        run_optimization.load_target_data()
        :param optimization_targets: Dictionary with optimization 
        targets,  coming from run_optimization.load_target_data()
        :param params: adjoint contraction paramters
        
        """
        

        
        self.bcs = bcs
        self.mesh_type = params["Patient_parameters"]["mesh_type"]
        self.solver_parameters = solver_parameters

        self.pressure = pressure
            
        self.target_params = params["Optimization_targets"]
        self.params = params

        self.meshvol = Constant(assemble(Constant(1.0)*dx(solver_parameters["mesh"])),
                                name = "mesh volume")

        self.regularization = optimization_targets.pop("regularization", None)
        # Initialize target functions
        for target in optimization_targets.values():
            target.set_target_functions()

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
        
        return "\nFuncional = {}\n".format((len(self.opt_weights.keys())*" {{}}*I_{} +").\
                                            format(*self.opt_weights.keys())[:-1].\
                                            format(*self.opt_weights.values()))
        
        

    def solve_the_forward_problem(self, phm, annotate = False, phase = "passive"):
        """Solve the forward model

        :param annotate: 
        :param phm: A heart problem instance
        :param phase: Which phase of the cycle, options: ['passive', 'active']
        :returns: A dictionary with the results
        :rtype: dict

        """
        

        # Set the functional value for each target to zero
        for key,val in self.target_params.iteritems():
            if val: self.optimization_targets[key].reset()
                
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
       
        # Get the functional value of each term in the functional
        func_lst = []
        for key,val in self.target_params.iteritems():
            if val:
                func_lst.append(self.opt_weights[key]*self.optimization_targets[key].get_functional())

        # Collect the terms in the functional
        functional = list_sum(func_lst)
              
       
        if phase == "active":
            # Add regulatization term to the functional
            m = phm.solver.parameters['material'].gamma

            functional += self.regularization.get_functional(m)
            reg_term =  self.regularization.get_value()

        else:
            # Add the initial state to the recording
            functionals_time.append(functional*dt[0.0])
            reg_term = 0.0

        
        for it, p in enumerate(self.bcs["pressure"][1:], start=1):

            sol = phm.next()
           
            for key,val in self.target_params.iteritems():

                if val:
            
                    self.optimization_targets[key].next_target(it, annotate=annotate)
                    self.optimization_targets[key].assign_simulated(split(sol)[0])
                    self.optimization_targets[key].assign_functional()
                    self.optimization_targets[key].save()
                    
            self.regularization.save()
            
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
    
    def _print_finished_report(self, forward_result):

        # from IPython import embed; embed()
        # exit()
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
        
        fr = {"optimization_targets": self.optimization_targets,
              "regularization": self.regularization,
              "states": self.states,
              "bcs": self.bcs,
              "total_functional": list_sum(functionals_time),
              "func_value": sum(functional_values)}
             
        return fr


class ActiveForwardRunner(BasicForwardRunner):
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
                                       pressure,
                                       params,
                                       annotate = False)
	
        # logger.debug("\nVolume before pressure change: {:.3f}".format(self.cphm.get_inner_cavity_volume()))
        self.cphm.increase_pressure()
        # logger.debug("Volume after pressure change: {:.3f}".format(self.cphm.get_inner_cavity_volume()))

    def __call__(self, m,  annotate = False):
	    
        logger.info("Evaluating model")
        # Take small steps with gamma until we have only one point left
        # We do not want to record this as we only want to optimize the final value
        logger.debug(Text.yellow("Stop annotating"))
        parameters["adjoint"]["stop_annotating"] = True
        try:
            # Try to step up gamma to the given one
            logger.debug("Try to step up gamma")
            w_old = self.cphm.get_state()
            gamma_old = self.gamma_previous.copy()
            self.cphm.next_active(m, self.gamma_previous.copy())
	    
        except StopIteration:
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
                
                

            # If stepping up gamma fails, assign the previous gamma
            # and return a crash=True, so that the Reduced functional
            # knows that we were unable to step up gamma
            logger.debug(Text.yellow("Start annotating"))
            parameters["adjoint"]["stop_annotating"] = not annotate

            logger.debug("Assign the old state and old gamma")
            # Assign the old state
            self.cphm.solver.get_state().assign(w_old, annotate=annotate)
            # Assign the old gamma
            self.cphm.solver.parameters['material'].gamma.assign(m)
            self.gamma_previous.assign(gamma_old)

            # Solve the forward problem with the old gamma
            logger.debug("Solve the forward problem with the old gamma")
            forward_result = BasicForwardRunner.solve_the_forward_problem(self, self.cphm,
                                                                          annotate, "active")

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
            self.cphm.solver.get_state().assign(w, annotate=annotate)

            # Now we make the final solve
            self.cphm.solver.parameters['material'].gamma.assign(m)
            self.gamma_previous.assign(m)

            logger.debug("Solve the forward problem with the new gamma")
            # Relax on the convergence criteria in order to ensure convergence
            self.cphm.solver.parameters["solve"]["snes_solver"]['absolute_tolerance']*= 100
            self.cphm.solver.parameters["solve"]["snes_solver"]['relative_tolerance']*= 100
            forward_result = BasicForwardRunner.solve_the_forward_problem(self, self.cphm,
                                                                          annotate, "active")
            self.cphm.solver.parameters["solve"]["snes_solver"]['absolute_tolerance']*= 0.01
            self.cphm.solver.parameters["solve"]["snes_solver"]['relative_tolerance']*= 0.01

            return forward_result, False




class PassiveForwardRunner(BasicForwardRunner):
    def __init__(self, solver_parameters, pressure, 
                 bcs, optimization_targets,
                 params, paramvec):

        
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

    def __call__(self, m, annotate = False):

        self.paramvec.assign(m)
       

        npassive = sum([ not self.params["Optimization_parmeteres"][k] \
                     for k in ["fix_a", "fix_a_f", "fix_b", "fix_b_f"]])
    

        lst = ["fix_a", "fix_a_f", "fix_b", "fix_b_f"]
        if npassive == 1:
            fixed_idx = np.nonzero([not self.params["Optimization_parmeteres"][k] for k in lst])[0][0]
            par = lst[fixed_idx].split("fix_")[-1]
            if self.params["matparams_space"] == "regional":
                paramvec = self.paramvec.get_function()
            else:
                paramvec = self.paramvec
            setattr(self.solver_parameters["material"], par, paramvec)
        else:
            paramvec_split = split(self.paramvec)
            fixed_idx = np.nonzero([not self.params["Optimization_parmeteres"][k] for k in lst])[0]
            for it, idx in enumerate(fixed_idx):
                par = lst[idx].split("fix_")[-1]
                setattr(self.solver_parameters["material"], par, paramvec_split[it])
                
     
        phm = PassiveHeartProblem(self.bcs,
                                  self.solver_parameters,
                                  self.pressure)

        # Do an initial solve for the initial point
        parameters["adjoint"]["stop_annotating"] = True
        phm.solver.solve()
        parameters["adjoint"]["stop_annotating"] = not annotate
        
        forward_result = BasicForwardRunner.solve_the_forward_problem(self, phm, annotate, "passive")


        return forward_result, False