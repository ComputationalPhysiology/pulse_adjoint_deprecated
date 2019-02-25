#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
import math
import numpy as np
import collections
import dolfin
import dolfin_adjoint

from .adjoint_contraction_args import *

from .utils import Text, UnableToChangePressureExeption
from pulse.iterate import iterate, delist
from pulse import numpy_mpi

def create_mechanics_problem(solver_parameters):
    import pulse
    # from pulse import (MechanicsProblem, HeartGeometry, BoundaryConditions,
                       # NeumannBC, RobinBC, MarkerFunctions, Marker, CRLBasis)
    from pulse.material import HolzapfelOgden

    mfun = pulse.MarkerFunctions(ffun=solver_parameters['facet_function'],
                                 cfun=solver_parameters['mesh_function'])
    
    material = solver_parameters['material']
    microstructure = pulse.Microstructure(f0=material.f0,
                                          s0=material.s0,
                                          n0=material.f0)
    
    crl_basis = pulse.CRLBasis(c0=solver_parameters['crl_basis']['circumferential'],
                               r0=solver_parameters['crl_basis']['radial'],
                               l0=solver_parameters['crl_basis']['longitudinal'])
    
    geometry = pulse.HeartGeometry(mesh=solver_parameters['mesh'],
                                   markers=solver_parameters['markers'],
                                   marker_functions=mfun,
                                   microstructure=microstructure,
                                   crl_basis=crl_basis)

    neumann = []
    for i, n in enumerate(solver_parameters['bc']['neumann']):
        neumann.append(pulse.NeumannBC(traction=n[0],
                                       marker=n[1], name=f'neumann_{i}'))

    robin = []
    for i, n in enumerate(solver_parameters['bc']['robin']):
        robin.append(pulse.RobinBC(value=n[0], marker=n[1]))


    if hasattr(solver_parameters['bc']['dirichlet'], '__len__'):
        dirichlet = solver_parameters['bc']['dirichlet']
    else:
        dirichlet = (solver_parameters['bc']['dirichlet'],)

    bcs = pulse.BoundaryConditions(dirichlet=dirichlet,
                                   neumann=neumann,
                                   robin=robin)

    problem = pulse.MechanicsProblem(geometry, material, bcs)

    return problem
    

class BasicHeartProblem(collections.Iterator):
    """
    This is a basic class for the heart problem.
    """

    def __init__(self, bcs, solver_parameters, pressure):

        self._init_pressures(bcs["pressure"], pressure["p_lv"], "lv")
        self.p_lv.assign(dolfin_adjoint.Constant(float(self.lv_pressure[0])))

        if "p_rv" in pressure:
            self.has_rv = True
            self._init_pressures(bcs["rv_pressure"], pressure["p_rv"], "rv")
            self.p_rv.assign(dolfin_adjoint.Constant(float(self.rv_pressure[0])))
        else:
            self.has_rv = False

        # Mechanical solver Active strain Holzapfel and Ogden
        self.solver = create_mechanics_problem(solver_parameters)

    def increase_pressure(self):

        p_lv_next = next(self.lv_pressure_gen)
        if self.has_rv:
            p_rv_next = next(self.rv_pressure_gen)
            target = (p_lv_next, p_rv_next)
            control = (self.p_lv, self.p_rv)
        else:
            target = p_lv_next
            control = self.p_lv

        iterate(problem=self.solver,
                target=target,
                control=control, continuation=True)

    def get_state(self, copy=True):
        """
        Return a copy of the state
        """
        if copy:
            return self.solver.state.copy(deepcopy=True)
        else:
            return self.solver.state

    def get_gamma(self, copy=True):

        gamma = self.solver.material.activation
        if isinstance(gamma, dolfin_adjoint.Constant):
            return gamma

        if copy:
            return gamma.copy(True)
        else:
            return gamma

    def _init_pressures(self, pressure, p, chamber="lv"):

        setattr(self, "{}_pressure".format(chamber), pressure)
        setattr(self, "{}_pressure_gen".format(chamber), (p for p in pressure[1:]))
        setattr(self, "p_{}".format(chamber), p)

    def __next__(self):
        """Solve the system as it is
        """

        self.solver.solve()

        return self.get_state(False)


def get_mean(f):
    return numpy_mpi.gather_broadcast(f.vector().array()).mean()


def get_max(f):
    return numpy_mpi.gather_broadcast(f.vector().array()).max()


def get_max_diff(f1, f2):

    diff = f1.vector() - f2.vector()
    diff.abs()
    return diff.max()


class ActiveHeartProblem(BasicHeartProblem):
    """
    A heart problem for the regional contracting gamma.
    """

    def __init__(self, bcs, solver_parameters, pressure, params, annotate=False):

        passive_filling_duration = solver_parameters["passive_filling_duration"]
        self.acin = params["active_contraction_iteration_number"]
        fname = "active_state_{}.h5".format(self.acin)
        if os.path.isfile(fname):
            if dolfin.mpi_comm_world().rank == 0:
                os.remove(fname)

        BasicHeartProblem.__init__(self, bcs, solver_parameters, pressure)

        # Load the state from the previous iteration
        w_temp = dolfin_adjoint.Function(self.solver.state_space, name="w_temp")
        with dolfin.HDF5File(dolfin.mpi_comm_world(), params["sim_file"], "r") as h5file:

            # Get previous state
            if params["active_contraction_iteration_number"] == 0:
                it = (
                    passive_filling_duration
                    if params["unload"]
                    else passive_filling_duration - 1
                )
                group = "/".join(
                    [params["h5group"], PASSIVE_INFLATION_GROUP, "states", str(it)]
                )

            else:
                group = "/".join(
                    [
                        params["h5group"],
                        ACTIVE_CONTRACTION_GROUP.format(
                            params["active_contraction_iteration_number"] - 1
                        ),
                        "states",
                        "0",
                    ]
                )

            h5file.read(w_temp, group)

        self.solver.reinit(w_temp, annotate=annotate)
        self.solver.solve()

    def get_number_of_stored_states(self):

        fname = "active_state_{}.h5".format(self.acin)
        if os.path.isfile(fname):
            i = 0
            with dolfin.HDF5File(dolfin.mpi_comm_world(), fname, "r") as h5file:
                group_exist = h5file.has_dataset("0")
                while group_exist:
                    i += 1
                    group_exist = h5file.has_dataset(str(i))

            return i

        else:
            return 0

    def store_states(self, states, gammas):

        fname = "active_state_{}.h5".format(self.acin)
        file_mode = "a" if os.path.isfile(fname) else "w"
        key = self.get_number_of_stored_states()

        gamma_group = "{}/gamma"
        state_group = "{}/state"

        assert len(states) == len(
            gammas
        ), "Number of states does not math number of gammas"

        with dolfin.HDF5File(dolfin.mpi_comm_world(), fname, file_mode) as h5file:

            for (w, g) in zip(states, gammas):
                h5file.write(w, state_group.format(key))
                h5file.write(g, gamma_group.format(key))
                key += 1

    def load_states(self):

        fname = "active_state_{}.h5".format(self.acin)
        if not os.path.isfile(fname):
            return [], []

        nstates = self.get_number_of_stored_states()

        gamma_group = "{}/gamma"
        state_group = "{}/state"

        states = []
        gammas = []

        w = self.solver.state.copy(True)
        g = self.solver.material.activation.copy(True)

        with dolfin.HDF5File(dolfin.mpi_comm_world(), fname, "r") as h5file:

            for i in range(nstates):

                try:
                    h5file.read(w, state_group.format(i))
                    h5file.read(g, gamma_group.format(i))

                except:
                    logger.info("State {} does not exist".format(i))

                else:
                    states.append(w.copy(True))
                    gammas.append(g.copy(True))

        return states, gammas

    def next_active(self, gamma_current, gamma, assign_prev_state=True, steps=None):

        old_states, old_gammas = self.load_states()

        states, gammas = iterate(
            problem=self.solver,
            control=gamma,
            target=gamma_current,
            continuation=True,
            old_states=old_states,
            old_controls=old_gammas
        )
        # Store these gammas and states which can be used
        # as initial guess for the newton solver in a later
        # iteration
        self.store_states(states, gammas)

        if assign_prev_state:
            # Assign the previous state
            self.solver.reinit(states[-1])
            self.solver.material.activation.assign(gammas[-1])

        return self.get_state(False)


class PassiveHeartProblem(BasicHeartProblem):
    """
    Runs a biventricular simulation of the diastolic phase of the cardiac
    cycle. The simulation is driven by LV pressures and is quasi-static.
    """

    def __next__(self):
        """
        Increase the pressure and solve the system
        """

        self.increase_pressure()

        return self.get_state(False)
