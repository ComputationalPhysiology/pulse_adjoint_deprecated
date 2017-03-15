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
from iterate import iterate

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
       

    def increase_pressure(self):

        p_lv_next = self.lv_pressure_gen.next()
        if self.has_rv:
            p_rv_next = self.rv_pressure_gen.next()
            target_pressure = (p_lv_next, p_rv_next)
            pressure = {"p_lv":self.p_lv, "p_rv":self.p_rv}
        else:
            target_pressure = p_lv_next
            pressure = {"p_lv":self.p_lv}

        iterate(self.solver, target_pressure, "pressure",
                pressure)
        
    
    def get_state(self, copy = True):
        """
        Return a copy of the state
        """
        if copy:
            return self.solver.get_state().copy(True)
        else:
            return self.solver.get_state()
        
    def get_gamma(self, copy =True):
        if copy:
            return self.solver.parameters["material"].gamma.copy(True)
        else:
            return self.solver.parameters["material"].gamma

            
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
        
        self.solver.solve()
	
        return self.get_state(False)



def get_mean(f):
    return gather_broadcast(f.vector().array()).mean()


def get_max(f):
    return gather_broadcast(f.vector().array()).max()

def get_max_diff(f1,f2):

    diff = f1.vector() - f2.vector()
    diff.abs()
    return diff.max()



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
                it = passive_filling_duration if params["unload"] else passive_filling_duration-1
                group = "/".join([params["h5group"],
                                  PASSIVE_INFLATION_GROUP,
                                  "states",
                                  str(it)])
                
            else:
                group = "/".join([params["h5group"],
                                  ACTIVE_CONTRACTION_GROUP.format(params["active_contraction_iteration_number"] - 1),
                                  "states", "0"])
                
            h5file.read(w_temp, group)

        self.solver.reinit(w_temp, annotate=annotate)
        # self.solver.get_state().assign(w_temp, annotate = annotate)
        self.solver.solve()
       
    def next_active(self, gamma_current, gamma, assign_prev_state=True, steps = None):

        gammas, states = iterate(self.solver, gamma_current, "gamma")

        if assign_prev_state:
            # Assign the previous state
            self.solver.reinit(states[-1])
            self.solver.parameters['material'].gamma.assign(gammas[-1])
        
        return self.get_state(False)





class PassiveHeartProblem(BasicHeartProblem):
    """
    Runs a biventricular simulation of the diastolic phase of the cardiac
    cycle. The simulation is driven by LV pressures and is quasi-static.
    """
    def next(self):
        """
        Increase the pressure and solve the system
        """
        
        self.increase_pressure()
        
        return self.get_state(False)



