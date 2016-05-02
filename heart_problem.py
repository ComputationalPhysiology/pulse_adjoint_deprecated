#!/usr/bin/env python
# Copyright (C) 2016 Henrik Finsberg
#
# This file is part of CAMPASS.
#
# CAMPASS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CAMPASS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CAMPASS. If not, see <http://www.gnu.org/licenses/>.

from dolfin import *
from dolfin_adjoint import *
from adjoint_contraction_args import *
import math
import numpy as np
from numpy_mpi import *
from utils import Text
import collections
from lvsolver import LVSolver



class BasicHeartProblem(collections.Iterator):
    """
    Runs a biventricular simulation of the diastolic phase of the cardiac
    cycle. The simulation is driven by LV and RV pressures and is quasi-static.
    """
    def __init__(self, pressure, solver_parameters, p_lv, 
                 endo_lv_marker, crl_basis, spaces):

        self._init_pressures(pressure, p_lv)
        
        self._init_measures_and_markers(endo_lv_marker, 
                                        solver_parameters)

        #Objects needed for Volume calculation
        self._init_strain_functions(spaces)
        
        self.crl_basis = crl_basis

        # Mechanical solver Active strain Holzapfel and Ogden
        self.solver = LVSolver(solver_parameters)
        
        self.p_lv.t = self.pressure[0]
     
	
    
    def next(self):

        out = self.increase_pressure()
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

            crash = False
            for p in pressures[1:]:
                self.p_lv.t = p

                out, crash = self.solver.solve()
                
                if crash:
                    logger.warning("Solver chrashed when increasing pressure from {} to {}".format(p_prev, p))
                    logger.warning("Take smaller steps")
                    nsteps *= 2
                    pressures = np.linspace(p_prev, p_next, nsteps)
                    break
                else:
                    p_prev = p

            
            converged = False if crash else True
                

        if nsteps >= n_max:
            raise RuntimeError("Unable to increase pressure")

        
        return out
    
    def get_state(self):
        return self.solver.get_state().copy(True)

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
        self.strainfield = Function(self.strainfieldspace, name = "Simulated StrainField")
        
        self.strainspace = spaces.strainspace
        self.strains = [Function(self.strainspace,
                                     name = "Simulated Strain_{}".format(i)) for i in STRAIN_REGION_NUMS]

        self.strain_u = TrialFunction(self.strainspace)
        self.strain_v = TestFunction(self.strainspace)

        self.strainfield_u = TrialFunction(self.strainfieldspace)
        self.strainfield_v = TestFunction(self.strainfieldspace)
        
       
    def _init_measures_and_markers(self, endo_lv_marker, solver_parameters):
        # Boundary markers
        ffun = solver_parameters["facet_function"]
        # Mesh
        self.mesh = solver_parameters["mesh"]
        # Surface measure
        self.ds = Measure("exterior_facet", subdomain_data = ffun, domain = self.mesh)(endo_lv_marker)
        # Volume measure
        self.dx = Measure("dx", subdomain_data = solver_parameters["mesh_function"],
                                domain = solver_parameters["mesh"])
        
        self.strain_markers = solver_parameters["mesh_function"]
        

    def _init_volume_forms(self):
        # Reference coordinates
        X = SpatialCoordinate(self.mesh)

        # Facet Normal 
        N = self.solver.parameters["facet_normal"]

        # Collect displacement u
        self.u = self.solver.get_u() 

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

                self.solver.parameters['material'].gamma.assign(g)

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



class ActiveHeartProblem(BasicHeartProblem):
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


        w_temp = Function(self.solver.get_state_space(), name = "w_temp")
        with HDF5File(mpi_comm_world(), params["sim_file"], 'r') as h5file:
        
            # Get previous regional gamma and state
            if params["active_contraction_iteration_number"] == 0:
                h5file.read(w_temp, PASSIVE_INFLATION_GROUP.format(params["alpha_matparams"]) + \
                            "/states/{}".format(passive_filling_duration - 1))
            else:
                h5file.read(w_temp, ACTIVE_CONTRACTION_GROUP.
                            format(params["alpha"],
                                   params["active_contraction_iteration_number"] - 1) + "/states/0")


        self.solver.get_state().assign(w_temp, annotate = annotate)
        BasicHeartProblem._init_volume_forms(self)
       
    def next(self):
        out = self.solver.solve()
	
        strains = self.project_to_strains(self.u)
        return out, strains

    
   
    
        
    
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


        # from IPython import embed; embed()
        
        # V = FunctionSpace(self.mesh, "CG", 1)
        #
        # g = Function(V, name = "g")
        g = Function(g_prev.function_space(), name = "g")
        # g.assign(g_prev)
        g.vector()[:] = g_prev.vector()[:]

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
                    self.solver.parameters['material'].gamma.assign(g_old)
                    # self.solver.get_state().assign(w_old)
                    raise StopIteration("Iteration have chrashed too many times")

                
                # Loop over the steps
                for i in range(1, nr_steps):
                
                    # Increment gamma
                    g.vector()[:] +=  dg.vector()[:]
                    # Assing the new gamma
                    self.solver.parameters['material'].gamma.assign(g)
                        
                    # try:
                    # Try to solve
                    logger.debug("\t{:.3f} \t{:.3f}".format(get_mean(g), get_max(g)))
                    out, crash = self.solver.solve()

                    # except RuntimeError as ex:
                    if crash:
                        # If that does not work increase the number of steps
                        logger.warning("Solver crashed. Reduce gamma step")
                        nr_steps *=2
                        
                        # Assign the previous gamma
                        g.assign(g_prev)
                        
                        dg.vector()[:] = 1./nr_steps * (gamma_current.vector()[:] 
                                                        - g_prev.vector()[:]) 
                        nr_crashes += 1

                        break

                    else:
                        g_prev.assign(g.copy(True))
                        
                        idx += 1
                        
                if i == nr_steps-1:
                    finished_stepping = True

            # All points up to the last point converged. 
            # Now check that the last point also converges.
            
            # Store the current solution
            w = self.get_state()

            self.solver.parameters['material'].gamma.assign(gamma_current)
            
            # try:
            logger.debug("\t{:.3f} \t{:.3f}".format(get_mean(gamma_current), 
                                                       get_max(gamma_current)))
            out, crash = self.solver.solve()
            # except RuntimeError:
            if crash:
                nr_steps *= 2
                logger.warning("\tFinal solve-step crashed. Reduce gamma step")
                g.assign(g_prev)
                dg.vector()[:] = 1./(nr_steps) * (gamma_current.vector()[:] - g_prev.vector()[:]) 

                finished_stepping = False
            else:
                # Assign the previous state
                self.solver.get_state().assign(w, annotate = False)
                self.solver.reinit(w)
                self.solver.parameters['material'].gamma.assign(g_prev)
              
                done = True
                
        


        return out




class PassiveHeartProblem(BasicHeartProblem):
    def __init__(self, pressure, solver_parameters, p_lv, 
                 endo_lv_marker, crl_basis, spaces):
       
        BasicHeartProblem.__init__(self, pressure, solver_parameters, p_lv, 
                                    endo_lv_marker, crl_basis, spaces)

        BasicHeartProblem._init_volume_forms(self)



