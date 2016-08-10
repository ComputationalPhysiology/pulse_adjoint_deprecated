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
from lvsolver import LVSolver

class BasicHeartProblem(collections.Iterator):
    """
    This is a basic class for the heart problem.
    """
    def __init__(self, pressure, solver_parameters, p_lv, 
                 endo_lv_marker, crl_basis, spaces):

        self._init_pressures(pressure, p_lv)
        
        self._init_measures_and_markers(endo_lv_marker, 
                                        solver_parameters)

        #Objects needed for Volume calculation
        self._init_strain_functions(spaces)
        
        # Basis function in the circumferential, 
        # radial and longitudinal direction
        self.crl_basis = crl_basis

        # Mechanical solver Active strain Holzapfel and Ogden
        self.solver = LVSolver(solver_parameters)
        
        # Start with the first pressure
        self.p_lv.t = self.pressure[0]
        
    
        self.base_bc_from_seg = solver_parameters["base_bc_y"] is not None

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
                out, crash = self.solver.solve()
                
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
        p_next = self.pressure_gen.next()
        p_prev = self.p_lv.t

        p_diff = abs(p_next - p_prev)
        if p_diff < DOLFIN_EPS:
            return self.solver.solve()[0]

        logger.debug("Increase pressure:  previous   next")
        logger.debug("\t            {:.3f}     {:.3f}".format(p_prev, p_next))

        converged = False
        # nsteps = max(2, int(math.ceil(p_diff/PRESSURE_INC_LIMIT)))
        # nsteps = np.ceil(abs((p_next - p_prev)/(p_prev+1)))
        nsteps = 2
        n_max = 100
        pressures = np.linspace(p_prev, p_next, nsteps)
        p = p_prev

        while not converged and nsteps < n_max:

            crash = False
            for p in pressures[1:]:
                self.p_lv.t = p
                
                logger.debug("\nSolve for pressure = {}".format(p))
                out, crash = self.solver.solve()
                
                
                if crash:
                    logger.warning("\nSolver chrashed when increasing pressure from {} to {}".format(p_prev, p))
                    logger.warning("Take smaller steps")
                    nsteps *= 2
                    pressures = np.linspace(p_prev, p_next, nsteps)
                    break
                else:
                    p_prev = p
                    # Adapt
                    nsteps = np.ceil(nsteps/1.5)
                    pressures = np.linspace(p_prev, p_next, nsteps)
                    converged = True if p == p_next else False
                    break
                    


        if nsteps >= n_max:
            
            raise UnableToChangePressureExeption("Unable to increase pressure")

        
        if self.base_bc_from_seg:
            self.set_direchlet_bc()

        
        return out
    
    def get_state(self):
        """
        Return a copy of the state
        """
        return self.solver.get_state().copy(True)

    

    def get_inner_cavity_volume(self):
        """
        Return the volume of left ventricular chamber
        """
        return assemble(self.vol_form)

    def _init_pressures(self, pressure, p_lv):
        self.pressure = pressure
        self.pressure_gen = (p for p in pressure[1:])
        self.p_lv = p_lv
        

    def _init_strain_functions(self, spaces):
        """
        Initialze spaces and functions used for strain calculations
        """
        
        
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
        """
        Load mesh, measures and boundary markers
        """
        # Boundary markers
        ffun = solver_parameters["facet_function"]
        # Mesh
        self.mesh = solver_parameters["mesh"]
        # Surface measure
        self.ds = Measure("exterior_facet", subdomain_data = ffun, domain = self.mesh)(endo_lv_marker)
        # Volume measure, with each index corresponding to a strain region
        self.dx = Measure("dx", subdomain_data = solver_parameters["mesh_function"],
                                domain = solver_parameters["mesh"])
        
        self.strain_markers = solver_parameters["mesh_function"]
        

    def _init_volume_forms(self):
        """
        UFL form form computing inner cavity volume
        """
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

    def project_to_strains(self, u):
        """
        In order for dolfin-adjoint to record that the
        strain functional changes during the opimization, 
        we need to solve an eaqation that makes the recording. 
        """

        # Take of the correct components of the displacement gradient
        gradu = grad(u)
        grad_u_diag = as_vector([inner(e,gradu*e) for e in self.crl_basis])

        # Solve for the strain fields
        # Somehow this does not work with LU solver when base is fixed according to the
        # segemental surfaces
        solve(inner(self.strainfield_u, self.strainfield_v)*dx == \
                  inner(self.strainfield_v, grad_u_diag)*dx,self.strainfield, solver_parameters={"linear_solver": "gmres"})
    
        # Solve for the regional strains
        for i in STRAIN_REGION_NUMS:
            solve(inner(self.strain_u, self.strain_v)*self.dx(i) == \
                  inner(self.strain_v, grad_u_diag)*self.dx(i), \
                  self.strains[i - 1])

        return self.strains

    def next(self):
        """
        Solve the system as it is
        """
        out = self.solver.solve()
	
        strains = self.project_to_strains(self.u)
        return out, strains



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
                 pressure,
                 solver_parameters,
                 p_lv,
                 endo_lv_marker,
                 crl_basis,
                 spaces,
                 passive_filling_duration, 
                 params,
                 annotate = False):
                   
        
        self.alpha = params["alpha"]
        self.passive_filling_duration = passive_filling_duration
       
        

        BasicHeartProblem.__init__(self, pressure, solver_parameters, p_lv, 
                                    endo_lv_marker, crl_basis, spaces)

        # Load the state from the previous iteration
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
        
        
        logger.debug("\n\tIncrement gamma...")
        logger.debug("\tMean \tMax")
        while not done:
            while not finished_stepping:

                if nr_crashes > MAX_NR_CRASH:
                    self.solver.parameters['material'].gamma.assign(g_old)
                    raise StopIteration("Iteration have chrashed too many times")

                
                # Loop over the steps
                for i in range(1, nr_steps):
                
                    # Increment gamma
                    g.vector()[:] +=  dg.vector()[:]
                    # Assing the new gamma
                    self.solver.parameters['material'].gamma.assign(g)
                        

                    # Try to solve
                    logger.debug("\t{:.3f} \t{:.3f}".format(get_mean(g), get_max(g)))

                    # Make the convergence criteria stricter so that it is more likely to converge
                    # when annotating is on
                    self.cphm.solver.parameters["solve"]["snes_solver"]['absolute_tolerance']*= 0.01
                    self.cphm.solver.parameters["solve"]["snes_solver"]['relative_tolerance']*= 0.01
                    out, crash = self.solver.solve()
                    self.cphm.solver.parameters["solve"]["snes_solver"]['absolute_tolerance']*= 100
                    self.cphm.solver.parameters["solve"]["snes_solver"]['relative_tolerance']*= 100

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
            out, crash = self.solver.solve()
            
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
    def __init__(self, pressure, solver_parameters, p_lv, 
                 endo_lv_marker, crl_basis, spaces):
       
        BasicHeartProblem.__init__(self, pressure, solver_parameters, p_lv, 
                                    endo_lv_marker, crl_basis, spaces)

        BasicHeartProblem._init_volume_forms(self)

        
    def next(self):
        """
        Increase the pressure and solve the system
        """
        
        out = self.increase_pressure()
        strains = self.project_to_strains(self.u)
        
        return out, strains



