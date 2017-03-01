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
import math
import numpy as np
from numpy_mpi import *
from utils import Text, UnableToChangePressureExeption
import collections
from lvsolver import LVSolver, SolverDidNotConverge

class BasicHeartProblem(collections.Iterator):
    """
    This is a basic class for the heart problem.
    """
    def __init__(self, bcs, solver_parameters, pressure):

        self._init_pressures(bcs["pressure"], pressure["p_lv"], "lv")

        self.p_lv.t = self.lv_pressure[0]
        
        if pressure.has_key("p_rv"):
            self.has_rv = True
            self._init_pressures(bcs["rv_pressure"], pressure["p_rv"], "rv")
            self.p_rv.t = self.rv_pressure[0]
        else:
            self.has_rv = False

        # Mechanical solver Active strain Holzapfel and Ogden
        self.solver = LVSolver(solver_parameters)
        

        if solver_parameters.has_key("base_bc_y"):
            self.base_bc_from_seg = solver_parameters["base_bc_y"] is not None
        else:
            self.base_bc_from_seg = False
        if self.base_bc_from_seg:
            self.solver.parameters["base_bc_y"].reset()
            self.solver.parameters["base_bc_z"].reset()
         
            

    def set_direchlet_bc(self):
        """
        Set Direclet BC at the endoring by an iterative process
        """
        # for i in range(self.solver.parameters["base_bc_y"].npoints):
        self.solver.parameters["base_bc_y"].next()
        self.solver.parameters["base_bc_z"].next()
        it = self.solver.parameters["base_it"]
        
        nsteps = 2
        ts = np.linspace(0,1, nsteps)
        _i = 0.0
        done = False

        # In case the solver chrashes when solving for the
        # Dirichlet BC we can set this BC in an iterative way. 
        while not done and nsteps < 20:
            for i in ts[1:]:

                logger.debug("Iterator for BC = {}".format(i))
                it.t = i
                try:
                    out = self.solver.solve()
                    
                except SolverDidNotConverge:
                    crash = True
                else:
                    crash = False
                    
                if crash:
                    logger.debug("Crashed when setting BC")
                    nsteps += 2
                    ts = np.linspace(_i, 1, nsteps)
                    break
                
                else:
                    _i = i
                    
                    if i == 1:
                        done = True
        

    def increase_pressure(self):
        """
        Step up the pressure to the next given
        pressure and solve the force-balance equations
        """


        p_lv_next = self.lv_pressure_gen.next()
        p_lv_prev = self.p_lv.t
        p_diff = abs(p_lv_next - p_lv_prev)

        if self.has_rv:
            p_rv_next = self.rv_pressure_gen.next()
            p_rv_prev = self.p_rv.t
            
            p_diff += abs(p_rv_next - p_rv_prev)
            
  
        if p_diff < DOLFIN_EPS:
            # return solver.solve()[0]
            return self.solver.get_state()

        head = "{:<20}\t{:<15}\t{:<15}".format("Increase pressure:",
                                               "previous (lv)",
                                               "next (lv)")
        line = " "*20+"\t{:<15}\t{:<15}".format(p_lv_prev,
                                                p_lv_next)
        if self.has_rv:
            head += "\t{:<15}\t{:<15}".format("previous (rv)",
                                               "next (rv)")
            line += "\t{:<15}\t{:<15}".format(p_rv_prev,
                                              p_rv_next)
            
        logger.debug(head)
        logger.debug(line)

        converged = False

        nsteps = max(np.rint(p_diff/0.4), 2)
        n_max = 100
        
        lv_pressures = np.linspace(p_lv_prev, p_lv_next, nsteps)
        p_lv = p_lv_prev
        
        if self.has_rv:
            rv_pressures = np.linspace(p_rv_prev, p_rv_next, nsteps)
            p_rv = p_rv_prev
            
    
        while not converged and nsteps < n_max:

            crash = False
            for it, p_lv in enumerate(lv_pressures[1:], start = 1):
                
                self.p_lv.t = p_lv
                
                if self.has_rv:
                    p_rv = rv_pressures[it]
                    self.p_rv.t = p_rv
                
                logger.debug("\nSolve for lv pressure = {}".format(p_lv))
                try:
                    out = self.solver.solve()
                except SolverDidNotConverge:
                    crash = True
                else:
                    crash = False
                
                
                if crash:
                    logger.warning("\nSolver chrashed when increasing pressure from {} to {}".format(p_lv_prev, p_lv))
                    logger.warning("Take smaller steps")
                    nsteps *= 2
                    lv_pressures = np.linspace(p_lv_prev, p_lv_next, nsteps)
                    if self.has_rv:
                        rv_pressures = np.linspace(p_rv_prev, p_rv_next, nsteps)
                    break
                else:
                    p_lv_prev = p_lv
                    
                    # Adapt
                    nsteps = np.ceil(nsteps/1.5)
                    lv_pressures = np.linspace(p_lv_prev, p_lv_next, nsteps)
                    if self.has_rv:
                        p_rv_prev = p_rv
                        rv_pressures = np.linspace(p_rv_prev, p_rv_next, nsteps)
                        
                    converged = True if p_lv == p_lv_next else False
                    break
                    


        if nsteps >= n_max:
            
            raise UnableToChangePressureExeption("Unable to increase pressure")

        
        if self.base_bc_from_seg:
            self.set_direchlet_bc()

        
        return out
    
    def get_state(self, copy = True):
        """
        Return a copy of the state
        """
        if copy:
            return self.solver.get_state().copy(True)
        else:
            return self.solver.get_state()

    def get_inner_cavity_volume(self):
        """
        Return the volume of left ventricular chamber
        """
        return assemble(self.vol_form)

    def _init_pressures(self, pressure, p, chamber = "lv"):

        setattr(self, "{}_pressure".format(chamber), pressure)
        setattr(self, "{}_pressure_gen".format(chamber),
                (p for p in pressure[1:]))
        setattr(self, "p_{}".format(chamber), p)

        
        

    def next(self):
        """Solve the system as it is
        """

        
        out = self.solver.solve()
	
        return out



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
    def __init__(self, bcs,
                 solver_parameters,
                 p_lv,
                 gamma_list):
        
        self.gamma_gen = (g for g in gamma_list)
        self.gamma = gamma_list[0]
        
        BasicHeartProblem.__init__(self, bcs, solver_parameters, p_lv)

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

                self.solver.parameters['material'].gamma.assign(g)

              
                out, crash = self.solver.solve()
                if crash:
                    logger.warning("Solver crashed. Reduce gamma step")
                    
                    nr_steps += 4
                    g.assign(g_prev)
                    dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] - g_prev.vector()[:]) 
                    logger.debug("DG vector max {}".format(dg.vector().max()))
                    
                    break

                else:
                    g_prev.assign(g)
                    
                        
                if i == nr_steps:
                    done = True

        self.gamma = gamma_current

        return BasicHeartProblem.next(self)



class ActiveHeartProblem(BasicHeartProblem):
    """
    A heart problem for the regional contracting gamma.
    """
    def __init__(self,
                 bcs,
                 solver_parameters,
                 pressure,
                 params,
                 annotate = False):
                   
        
        
        passive_filling_duration = solver_parameters["passive_filling_duration"]
    
        BasicHeartProblem.__init__(self, bcs, solver_parameters, pressure)

        # Load the state from the previous iteration
        w_temp = Function(self.solver.get_state_space(), name = "w_temp")
        with HDF5File(mpi_comm_world(), params["sim_file"], 'r') as h5file:
        
            # Get previous state
            if params["active_contraction_iteration_number"] == 0:
                group = "/".join([params["h5group"],
                                  PASSIVE_INFLATION_GROUP,
                                  "states",
                                  str(passive_filling_duration - 1)])
                
            else:
                group = "/".join([params["h5group"],
                                  ACTIVE_CONTRACTION_GROUP.format(params["active_contraction_iteration_number"] - 1),
                                  "states", "0"])
                
            h5file.read(w_temp, group)


        self.solver.get_state().assign(w_temp, annotate = annotate)
        self.solver.solve()
       
    
    def next_active(self, gamma_current, gamma, assign_prev_state=True, steps = None):
        """
        Step up gamma iteratively.
        """

        max_diff = get_max_diff(gamma_current, gamma)
        nr_steps = max(2, int(math.ceil(max_diff/GAMMA_INC_LIMIT))) if steps is None else steps

        logger.debug("\tGamma:    Mean    Max     max difference")
        logger.debug("\tPrevious  {:.3f}  {:.3f}    {:.3e}".format(get_mean(gamma), 
                                                                   get_max(gamma), 
                                                                   max_diff))
        logger.debug("\tNext      {:.3f}  {:.3f} ".format(get_mean(gamma_current), 
                                                          get_max(gamma_current)))

        # Step size for gamma
        dg = Function(gamma.function_space(), name = "dg")
        dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] - gamma.vector()[:])

        # Gamma the will be used in the iteration
        g = Function(gamma.function_space(), name = "g")
        g.assign(gamma)

        g_previous = gamma.copy()

        # Keep the old gamma
        g_old = gamma.copy()
        
        done = False
        finished_stepping = False

             
        # If the solver crashes n times it is probably stuck
        MAX_NR_CRASH = 5
        nr_crashes = 0
        
        
        logger.info("\n\tIncrement gamma...")
        logger.info("\tMean \tMax")
        while not done:
            while not finished_stepping:
                
                

                
                # Loop over the steps
                for i in range(1, nr_steps):
                
                    # Increment gamma
                    g.vector()[:] +=  dg.vector()[:]
                    # Assing the new gamma
                    self.solver.parameters['material'].gamma.assign(g)
                        

                    # Try to solve
                    logger.info("\t{:.3f} \t{:.3f}".format(get_mean(g), get_max(g)))

                    try:
                        
                        out = self.solver.solve()
                
                    except SolverDidNotConverge as ex:
                        if nr_crashes > MAX_NR_CRASH:
                            # Throw exception further
                            raise ex
                        else:
                            crash = True
                    else:
                        crash = False
 
                    if crash:
                        # If that does not work increase the number of steps
                        logger.warning("Solver crashed. Reduce gamma step")
                        nr_steps *=2

                        g.assign(g_previous)
                            
                        
                        dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] 
                                                        - g_previous.vector()[:]) 
                        nr_crashes += 1

                        break

                    else:
                        g_previous.assign(g.copy())

                        # Adapt
                        # nr_steps /= 1.5
                        # dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] 
                        #                                 - g_prev.vector()[:])
                        
                        
                        
                if nr_steps == 1 or i == nr_steps-1:
                    finished_stepping = True

            # All points up to the last point converged. 
            # Now check that the last point also converges.
            
            # Store the current solution
            w = self.get_state()
            

            self.solver.parameters['material'].gamma.assign(gamma_current)
            
            
            logger.debug("\t{:.3f} \t{:.3f}".format(get_mean(gamma_current), 
                                                       get_max(gamma_current)))
            out = self.solver.solve()
      
            
            if crash:
                nr_steps *= 2
                logger.warning("\tFinal solve-step crashed. Reduce gamma step")
                g.assign(g_previous)
                dg.vector()[:] = 1./(nr_steps) * (gamma_current.vector()[:] - g_previous.vector()[:]) 

                finished_stepping = False
            else:
                if assign_prev_state:
                    # Assign the previous state
                    self.solver.get_state().assign(w, annotate = False)
                    self.solver.reinit(w)
                    self.solver.parameters['material'].gamma.assign(g_previous)
              
                done = True

        return out




class PassiveHeartProblem(BasicHeartProblem):
    """
    Runs a biventricular simulation of the diastolic phase of the cardiac
    cycle. The simulation is driven by LV pressures and is quasi-static.
    """
    def next(self):
        """
        Increase the pressure and solve the system
        """
        
        out = self.increase_pressure()
        
        return out



