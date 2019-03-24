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
"""
Unloading will find the reference geometry.
Here we also want to match the volumes or
strains and estimate the material parameteres
based on this
"""

__author__ = "Henrik Finsberg (henriknf@simula.no)"

import numpy as np
import dolfin as df

from pulse import HeartGeometry
from pulse.numpy_mpi import *
from pulse.unloader import *

from .utils import *

from ..setup_optimization import (
    setup_adjoint_contraction_parameters,
    make_control,
    setup_simulation,
    check_patient_attributes,
    make_solver_parameters
)
from ..run_optimization import run_passive_optimization_step, solve_oc_problem, store
from ..heart_problem import create_mechanics_problem



class UnloadedMaterial(object):
    """
    This class finds the unloaded cofiguration assuming 
    that the given geometry is loaded with `p_geo`.
    It iteratively estimate material parameters and unload 
    the geometry until the relative difference between the 
    current and previous volumes of the referece configuration
    is less than given tolerace

    This method is similar to the one described in [1].

    Parameters
    ----------
    
    geometry_index : int
        Index from which the given geometry correspond to in the 
        given data.    
    pressure : list
        List of pressure used for esimating material paramteters, the first
        entry in the list begin the same as the pressure in the geometry (`p_geo`).
        If BiV provide a list of tuples
    volumes : list
        List of volumes used for estimating material parameters, each volume corresponding
        to the volume at the given pressure in the pressure list. The offset between the 
        first volume and the volume in the geometry will be subtracted. 
    params : dict
        Application parameters from pulse_adjoint.setup_parameters.setup_adjoint_contraction_parameters.
        Used to setup the solver paramteteres. Note that the path to the original mesh should be in this
        dictionary, with key `Patient_parameters/mesh_path`. The output file will be saved to `sim_file`        
    method : str
        Which method to use for unloading. Options = ['fixed_point', 'raghavan', 'hybrid'].
        Default = 'hybrid'. For more info see :func`unloader.py`.
    tol : float
        Relative tolerance for difference in reference volume. Default = 5%.
    maxiter : int
        Maximum number of iterations of unloading/estimate parameters.
    unload_options: dict
        More info see :func`unloader.py`.

    Reference
    ---------
    .. [1] Nikou, Amir, et al. "Effects of using the unloaded configuration in predicting 
           the in vivo diastolic properties of the heart." Computer methods in biomechanics 
           and biomedical engineering 19.16 (2016): 1714-1720.

    """

    def __init__(
        self,
        geometry_index,
        pressures,
        volumes,
        params,
        paramvec,
        initial_guess=None,
        method="hybrid",
        tol=0.05,
        maxiter=10,
        continuation=True,
        unload_options={"maxiter": 10, "tol": 1e-2, "regen_fibers": True},
        optimize_matparams=True,
    ):

        p0 = pressures[0]
        self.it = 0
        self.is_biv = isinstance(p0, tuple) and len(p0) == 2
        self.params = params
        self.initial_guess = initial_guess
        self.continuation = continuation
        self.optimize_matparams = optimize_matparams

        self.geometry_index = geometry_index
        self.calibrate_data(volumes, pressures)

        self._backward_displacement = None
        self.unload_options = unload_options

        self._paramvec = paramvec.copy(deepcopy=True)

        # 5% change
        self.tol = tol
        self.maxiter = maxiter

        if method == "fixed_point":
            self.MeshUnloader = FixedPointUnloader
        elif method == "raghavan":
            self.MeshUnloader = RaghavanUnloader
        else:
            methods = ["fixed_point", "raghavan"]
            msg = "Unknown unloading algorithm {}. ".format(
                method
            ) + "Possible values are {}".format(methods)
            raise ValueError(msg)

        msg = (
            "\n\n"
            + " Start Unloaded Material Estimation  ".center(72, "#")
            + "\n\n\tgeometry_index = {}\n".format(geometry_index)
            + "\tpressures = {}\n".format(self.pressures)
            + "\tvolumes = {}\n".format(self.volumes)
            + "\tUnloading algorithm = {}\n".format(method)
            + "\ttolerance = {}\n".format(tol)
            + "\tmaxiter = {}\n".format(maxiter)
            + "\tcontinuation= {}\n\n".format(continuation)
            + "".center(72, "#")
            + "\n"
        )
        logger.info(msg)

        self.it = -1

    @property
    def geometry(self):
        return HeartGeometry.from_file(
            h5name=self.params["Patient_parameters"]["mesh_path"],
            h5group=self.params["Patient_parameters"]["mesh_group"],
        )
        

    def calibrate_data(self, volumes, pressures):

        if self.geometry.is_biv:
            v_lv = self.geometry.cavity_volume()
            v_lv_offset = v_lv - np.array(volumes).T[0][self.geometry_index]
            lv_volumes = np.add(np.array(volumes).T[0], v_lv_offset).tolist()
            logger.info("LV volume offset: {} ml".format(v_lv_offset))

            v_rv = self.geometry.cavity_volume(chamber="rv")
            v_rv_offset = v_rv - np.array(volumes).T[1][self.geometry_index]
            rv_volumes = np.add(np.array(volumes).T[1], v_rv_offset).tolist()
            logger.info("RV volume offset: {} ml".format(v_rv_offset))

            self.volumes = list(zip(lv_volumes, rv_volumes))

        else:

            v_lv = self.geometry.cavity_volume()
            v_lv_offset = v_lv - np.array(volumes).T[0]
            lv_volumes = np.add(np.array(volumes), v_lv_offset).tolist()
            logger.info("LV volume offset: {} ml".format(v_lv_offset))

            self.volumes = lv_volumes

        self.pressures = np.array(pressures).tolist()
        self.p_geo = self.pressures[self.geometry_index]

    def unload(self):

        paramvec, gamma, matparams = make_control(self.params, self.geometry)

        if self.it == 0 and self.initial_guess:
            assign_to_vector(
                paramvec.vector(), gather_broadcast(self.initial_guess.vector().array())
            )

        if self.it > 0:
            logger.info("Load control parmeters")
            load_material_parameter(self.params["sim_file"], str(self.it - 1), paramvec)

        if self.it > 1 and self.continuation:
            continuation_step(self.params, self.it, paramvec)

        logger.info(
            (
                "Value of control parameters = "
                + "{}".format(gather_broadcast(paramvec.vector().array()))
            )
        )

        self.params["phase"] = "unloading"

        params, p_expr = make_solver_parameters(
                self.params, self.geometry, matparams
            )

        problem = create_mechanics_problem(params)

        unloader = self.MeshUnloader(
            problem=problem,
            pressure=self.p_geo,
            h5name=self.params["sim_file"],
            options=self.unload_options,
            h5group=str(self.it),
            merge_control=self.params["merge_passive_control"],
        )
        # from IPython import embed; embed()
        # exit()
        unloader.unload()
        new_geometry = unloader.unloaded_geometry
        backward_displacement = unloader.backward_displacement

        group = "/".join([str(self.it), "unloaded"])
  
        new_geometry.save(
            h5name=self.params["sim_file"],
            h5group=group,
            other_functions=dict(backward_displacement=backward_displacement)
        )

        new_geometry.save(
            h5name=self.params["sim_file"],
            h5group="unloaded"
        )

        return HeartGeometry.from_file(
            h5name=self.params["sim_file"],
            h5group=group,
            comm=new_geometry.mesh.mpi_comm()
        )

    def get_backward_displacement(self):

        u = df.Function(df.VectorFunctionSpace(self.geometry.mesh, "CG", 1))

        group = "/".join([str(self.it), "unloaded", "backward_displacement"])

        with df.HDF5File(df.mpi_comm_world(), self.params["sim_file"], "r") as h5file:
            h5file.read(u, group)

        return u

    def get_unloaded_geometry(self):

        group = "/".join([str(self.it), "unloaded"])
        try:
            return HeartGeometry.from_file(h5name=self.params["sim_file"], h5grop=group)
        except IOError:
            msg = (
                "No unloaded geometry found {}:{}".format(
                    self.params["sim_file"], group
                )
                + "\nReturn original geometry."
            )
            logger.warning(msg)
            return self.geometry

    def get_optimal_material_parameter(self):

        paramvec = self._paramvec.copy(deepcopy=True)
        try:
            group = "/".join([str(self.it - 1), "passive_inflation", "optimal_control"])
            with df.HDF5File(
                df.mpi_comm_world(), self.params["sim_file"], "r"
            ) as h5file:
                h5file.read(paramvec, group)
            logger.info(
                "Load material parameter from {}:{}".format(
                    self.params["sim_file"], group
                )
            )
        except:
            logger.info("Could not open and read material parameter")

            # from IPython import embed; embed()
            # exit()
        return paramvec

    def get_loaded_volume(self, chamber="lv"):

        geo = self.get_unloaded_geometry()
        V = df.VectorFunctionSpace(geo.mesh, "CG", 2)
        u = df.Function(V)
        try:
            group = "/".join([str(self.it), "passive_inflation", "displacement", "1"])
            with df.HDF5File(
                df.mpi_comm_world(), self.params["sim_file"], "r"
            ) as h5file:
                h5file.read(u, group)
            logger.info(
                "Load displacement from {}:{}".format(self.params["sim_file"], group)
            )
        except:
            logger.info("Could not open and read displacement")

        return get.cavity_volume(chamber=chamber, u=u)

    def estimate_material(self):

        if self.it >= 0:
            group = "/".join([str(self.it), "unloaded"])
            logger.info(
                "Load geometry from {}:{}".format(self.params["sim_file"], group)
            )
            patient = HeartGeometry.from_file(self.params["sim_file"], group)

        else:
            patient = self.geometry

        patient.original_geometry = self.geometry.mesh

        patient.passive_filling_duration = len(self.pressures)
        if self.geometry.is_biv:
            patient.pressure = np.array(self.pressures).T[0]
            patient.volume = np.array(self.volumes).T[0]
            patient.RVP = np.array(self.pressures).T[1]
            patient.RVV = np.array(self.volumes).T[1]
            self.params["Patient_parameters"]["mesh_type"] = "biv"

        else:
            patient.pressure = self.pressures
            patient.volume = self.volumes
            self.params["Patient_parameters"]["mesh_type"] = "lv"

        self.params["h5group"] = str(self.it)

        measurements, solver_parameters, pressure, paramvec = setup_simulation(
            self.params, patient
        )

        if self.it > 0 or self.initial_guess:

            p_tmp = df.Function(paramvec.function_space())

            if self.it == 0:
                assign_to_vector(
                    p_tmp.vector(),
                    gather_broadcast(self.initial_guess.vector().array()),
                )
            else:
                # Use the previos value as initial guess
                p_tmp = df.Function(paramvec.function_space())
                load_material_parameter(
                    self.params["sim_file"], str(self.it - 1), p_tmp
                )

                if self.it > 1 and self.continuation:
                    continuation_step(self.params, self.it, p_tmp)

            paramvec.assign(p_tmp)

        logger.info(
            (
                "Value of control parameters = "
                + "{}".format(gather_broadcast(paramvec.vector().array()))
            )
        )

        rd, paramvec = run_passive_optimization_step(
            self.params, patient, solver_parameters, measurements, pressure, paramvec
        )

        res = solve_oc_problem(self.params, rd, paramvec, return_solution=True)
        return res

    def exist(self, key="unloaded"):

        import h5py
        from pulse_adjoint.utils import Text

        group = "/".join([str(self.it), key])
        with h5py.File(self.params["sim_file"]) as h5file:
            exist = group in h5file

        MPI.barrier(mpi_comm_world())
        if exist:
            logger.info(
                Text.green(
                    "{}, iteration {} - {}".format(
                        key, self.it, "fetched from database"
                    )
                )
            )
        else:
            logger.info(Text.blue("{}, iteration {} - {}".format(key, self.it, "Run")))
        return exist

    def copy_passive_inflation(self):

        import h5py

        group = "/".join([str(self.it), "passive_inflation"])
        if mpi_comm_world().rank == 0:
            with h5py.File(self.params["sim_file"]) as h5file:

                if not "passive_inflation" in h5file:
                    h5file.copy(group, "passive_inflation")

        MPI.barrier(df.mpi_comm_world())

    def compute_residual(self, it):

        if self.it > 0:

            group1 = "/".join([str(self.it - 1), "unloaded"])
            geo1 = HeartGeometry.from_file(self.params["sim_file"], group1)

            group2 = "/".join([str(self.it), "unloaded"])
            geo22 = HeartGeometry.from_file(self.params["sim_file"], group2)

            vol1_lv = geo1.cavity_volume()
            vol2_lv = geo2.cavity_volume()
            lv = abs(vol1_lv - vol2_lv) / vol1_lv

            if geo1.is_biv:
                vol1_rv = geo1.cavity_volume(chamber="rv")
                vol2_rv = geo2.cavity_volume(chamber="rv")
                rv = (vol1_rv - vol2_rv) / vol2_rv
            else:
                rv = 0.0

            return max(lv, rv)

        else:
            return np.inf

    def update_function_to_new_reference(self, fun, u, mesh=None):
        """
        Assume given function lives on the original
        geometry, and you want to find the function
        on the new referece geometry.
        Since the new referece geometry is topologically 
        equal to the old reference, the vector within the
        functions should be identical.

        Note that this is only relevant for functions of 
        rank 1, i.e vectors. 
        """
        if mesh is None:
            geo = self.get_unloaded_geometry()
            mesh = geo.mesh

        return update_vector_field(
            fun, mesh, u, str(fun), normalize=True, regen_fibers=False
        )

    def unload_material(self, patient=None):

        err = np.inf
        res = None

        while self.it < self.maxiter and err > self.tol:

            df.parameters["adjoint"]["stop_annotating"] = True
            if not self.exist("unloaded"):
                patient = self.unload()
            else:
                patient = self.get_unloaded_geometry()

            err = self.compute_residual(self.it)
            logger.info("\nCurrent residual:\t{}".format(err))

            df.parameters["adjoint"]["stop_annotating"] = False
            if not self.exist("passive_inflation"):
                res = self.estimate_material()

            self.it += 1

            if not self.optimize_matparams:
                break

        self.it -= 1

        if res is None:
            assert (
                self.it >= 0
            ), "You need to perform at least one iteration with unloading"
            self.copy_passive_inflation()
        else:
            # Store optimization results
            res[0]["h5group"] = ""
            store(*res)
