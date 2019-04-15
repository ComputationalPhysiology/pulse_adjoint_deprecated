#!/usr/bin/env python
"""
This script contains the container class that loads the results, 
and plot the results easily.
"""
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
from scipy import stats
import os, yaml
from ..setup_optimization import (
    setup_adjoint_contraction_parameters,
    setup_general_parameters,
    RegionalParameter,
)
from .args import *

from . import load, plot, utils, latex_utils, tables, vtk_utils


class PostProcess(object):
    """
    Initialize base class for postprocessing
    
    You need to provide three separate files.
    One HDF file (`fname`) containing the results.
    One HDF file (`geoname`) containing patient data.
    One yaml file (`pname`) containing adjoint contraction paramteters.

    All files should be organized with unque keys. 
    For instace, if you have two patients, e.g CRT and HC
    then two possible keys are 'CRT' and 'HC', and the results
    in `fname` should be orgranised as follows:
    
    ```
    fname
       CRT
          results for CRT
       HC
          results for HC
    ```
    and similarly for the other files.

    Parameters
    ----------
        
    fname: str
        Name of file where the results is stored. 
        See `load.get_data`
    geoname: str
        Name of file where the patient data is stored. 
        See `load.save_patient_to_h5`
    pname: str
        Name of file where the adjoint contraction parameters are stored. 
        See `load.save_parameters`
    outdir: str
        Directory where to save the results
    recompute: bool
        The data is stored in a yaml file and 
        will be loaded if it exist and recompute is False. 
        If recompute is True, then the feature will be recomputed
    
    """

    def __init__(self, fname, geoname, pname, outdir, tmp_dir=None, recompute=False):

        logger.info("Load file {}".format(fname))

        self._load_data(fname)
        self._fname = fname
        self._geoname = geoname
        self._pname = pname
        self.set_feature_keys()

        self._recompute = recompute
        self._outdir = outdir
        self._keys = list(self._data.keys())
        # self._mesh_path = mesh_path
        # self._pressure_path = pressure_path
        # self._echo_path = echo_path
        # self._params = params
        # self._simdir = simdir

        if tmp_dir is None:
            self._tmp_resdir = "/".join(
                [os.path.abspath(os.path.dirname(fname)), "tmp_results"]
            )
        else:
            self._tmp_resdir = tmp_dir

        if not os.path.exists(self._tmp_resdir):
            os.makedirs(self._tmp_resdir)

        self._results = {}
        self._features = {}

    def _load_data(self, fname):
        """Load data

        :param fname: path to datafile

        """

        self._data = load.load_dict_from_h5(fname)

    def _data_exist(self, patient_name, key):

        name = "_".join([patient_name, key]) + ".yml"
        return os.path.isfile("/".join([self._tmp_resdir, name]))

    def _load_tmp_results(self, patient_name, key, d):

        name = "_".join([patient_name, key]) + ".yml"
        path = "/".join([self._tmp_resdir, name])

        with open(path, "rb") as parfile:
            data = yaml.load(parfile)

        d.update(**data)

    def _save_tmp_results(self, patient_name, key, d):

        name = "_".join([patient_name, key]) + ".yml"
        path = "/".join([self._tmp_resdir, name])

        def listize(dic):
            d1 = {}
            for k, v in list(dic.items()):
                if isinstance(v, dict):
                    d1[k] = listize(v)
                elif isinstance(v, np.ndarray):
                    d1[k] = v.tolist()
                else:
                    d1[k] = v
            return d1

        d1 = listize(d)
        with open(path, "wb") as f:
            yaml.dump(d1, f, default_flow_style=False)

    def _update_results(self, data):
        """ Update results
        """

        if len(list(data.keys())) == 0:
            return

        if len(list(self._results.keys())) == 0:
            self._results.update(**data)
        else:
            utils.update_nested_dict(self._results, data)

    def _update_features(self, data):
        """ Update results
        """
        if len(list(self._features.keys())) == 0:
            self._features.update(**data)
        else:
            utils.update_nested_dict(self._features, data)

    def _set_matparams(self, patient, patient_name):

        params = self._params

        matlst = list(params["Fixed_parameters"].keys())

        if params["matparams_space"] == "regional":
            sfun = merge_control(patient, params["merge_passive_control"])
            paramvec = RegionalParameter(sfun)
        else:
            family, degree = params["matparams_space"].split("_")
            paramvec = dolfin.Function(
                dolfin.FunctionSpace(patient.mesh, family, int(degree)),
                name="matparam vector",
            )

        fixed_idx = np.nonzero([not params["Fixed_parameters"][k] for k in matlst])[0][
            0
        ]
        par = matlst[fixed_idx]

        if "material_parameters" in self._data[patient_name]:
            paramvec.vector()[:] = self._data[patient_name]["material_parameters"][par]

            matparams = {
                k: v[0]
                for k, v in self._data[patient_name]["material_parameters"].items()
            }
            matparams[par] = paramvec

            if not isinstance(params, dict):
                params = params.to_dict()
            params["Material_parameters"].update(**matparams)

        self._params = params
        return params

    def get_average_valve_times(self):
        """Get the percentage  average value of the time
        of the different valvular events starting at mvc

        Example

          mvc: 0%, avo: 5% , avc: 40%, mvo: 50%, 

        :returns: 
        :rtype: 

        """

        if self._valve_times is None:
            return None

        avo, avc, mvo = [], [], []

        for patient_name, v in self._valve_times.items():
            if "num_points" not in v:
                continue
            n = v["num_points"]
            echo_valve_times = v  # ["echo_valve_time"]
            pfb = v["passive_filling_begins"]

            avo.append(
                ((echo_valve_times["avo"] - echo_valve_times["mvc"]) % n) / float(n)
            )
            avc.append(
                ((echo_valve_times["avc"] - echo_valve_times["mvc"]) % n) / float(n)
            )
            mvo.append(
                ((echo_valve_times["mvo"] - echo_valve_times["mvc"]) % n) / float(n)
            )

        d = {"mvc": 0.0, "avo": np.mean(avo), "avc": np.mean(avc), "mvo": np.mean(mvo)}
        return d

    def _setup_compute(self, patient_name):

        self._label_key = r"Patient"

        # params = self._params
        # simdir = self._simdir
        # mesh_path = self._mesh_path
        # pressure_path = self._pressure_path
        # echo_path = self._echo_path

        logger.info("\nProcess data for patient {}".format(patient_name))

        self._params = load.load_parameters(self._pname, patient_name)
        patient = load.load_patient_data(self._geoname, patient_name)
        params = self._set_matparams(patient, patient_name)
        patient = load.load_measured_strain_and_volume(patient, params)

        from pulse_adjoint.setup_optimization import check_patient_attributes

        check_patient_attributes(patient)

        return params, patient

    def get_data(self, patient_name):

        val = self._data[patient_name]

        for k in ["displacements", "gammas", "states"]:
            if k not in val:
                val[k] = {}

        return val

    def get_meshvolumes(self, patient_name):
        """
        Return a dictionary with the volumes of each segement
        in the mesh.

        Parameters
        ----------

        patient_name : str
            The key/name of patient in the results dictionary.

        Returns
        -------

        meshvols : dict
             A dictionary with the volumes of each segement in the mesh.

        """
        patient = load.load_patient_data(self._geoname, patient_name)

        dx = dolfin.Measure("dx", domain=patient.mesh, subdomain_data=patient.sfun)
        meshvols = {}
        for r in set(patient.sfun.array()):
            meshvols[r] = dolfin.assemble(dolfin.Constant(1.0) * dx(int(r)))

        return meshvols

    def compute(self, *args):
        """
        Compute features that can be plotted later.
        Possible inputs are:

        * 'volume'
        * 'strain'
        * 'mean_gamma'
        * 'regional_gamma'
        * 'measurements'
        * 'emax'
        * 'geometric_distance'
        * 'time_varying_elastance'
        * 'vtk_simulation'
        * 'cardiac_work:{case}:{work_pair}', where case is 
           "full", "comp_fiber" or "comp_long", and work_pair 
           is "SE", "SEdev" "PF" or "pgradu"

        """

        if len(args) == 0:
            logger.info("Nothing to compute. Return")
            return

        setup_general_parameters()

        data = {}
        self._passive_filling_duration = {}
        for patient_name in self._keys:

            val = self.get_data(patient_name)

            data[patient_name] = {}

            params, patient = self._setup_compute(patient_name)
            self._passive_filling_duration[
                patient_name
            ] = patient.passive_filling_duration

            def compute_echo_work(patient, params):
                strain = {}
                load.load_measured_strain(strain, patient, "measured_strain")
                try:
                    # Not all the patients have meausred work
                    measured_work = {
                        k: np.transpose(v)[2].tolist() for k, v in patient.work.items()
                    }
                except:
                    # Compute work as how it would be measured from the strains
                    measured_work = {}
                    pressure = (
                        patient.pressure[1:] if params["unload"] else patient.pressure
                    )
                    for r, s in strain["measured_strain"]["longitudinal"].items():

                        measured_work[r] = utils.compute_cardiac_work_echo(
                            pressure, s, flip=True
                        )

                return measured_work

            if "measurements" in args:

                d = {}

                d["measured_volume"] = patient.volume
                d["pressure"] = patient.pressure

                load.load_measured_strain(d, patient, "measured_strain")

                if hasattr(patient, "rv_volume"):
                    d["measured_volume_rv"] = patient.rv_volume  # \
                    # + patient["data"]["rv_volume_offset"]
                    d["rv_pressure"] = patient.rv_pressure  # \
                    # + patient["data"]["rv_pressure_offset"]

                d["measured_work"] = compute_echo_work(patient, params)
                data[patient_name].update(**d)

            if "volume" in args:

                if self._data_exist(patient_name, "volume") and not self._recompute:
                    self._load_tmp_results(patient_name, "volume", data[patient_name])

                else:
                    logger.info("Compute volume")

                    d = {}

                    # LV
                    d["measured_volume"] = patient.volume
                    d["simulated_volume"] = utils.get_volumes(
                        val["displacements"],
                        patient,
                        "lv",
                        self._params["volume_approx"],
                    )
                    d["pressure"] = patient.pressure

                    # RV
                    if hasattr(patient, "rv_volume"):
                        d["measured_volume_rv"] = patient.rv_volume

                        d["simulated_volume_rv"] = utils.get_volumes(
                            val["displacements"],
                            patient,
                            "rv",
                            self._params["volume_approx"],
                        )

                        d["rv_pressure"] = patient.rv_pressure

                    self._save_tmp_results(patient_name, "volume", d)
                    data[patient_name].update(**d)

            if "strain" in args:
                if self._data_exist(patient_name, "strain") and not self._recompute:
                    self._load_tmp_results(patient_name, "strain", data[patient_name])

                else:

                    logger.info("Compute strain")
                    d = {}
                    d["simulated_strain"] = utils.get_regional_strains(
                        val["displacements"], patient, **params
                    )

                    load.load_measured_strain(d, patient, "measured_strain")

                    self._save_tmp_results(patient_name, "strain", d)
                    data[patient_name].update(**d)

            if "time_varying_elastance" in args:

                if (
                    self._data_exist(patient_name, "time_varying_elastance")
                    and not self._recompute
                ):
                    self._load_tmp_results(
                        patient_name, "time_varying_elastance", data[patient_name]
                    )

                else:
                    logger.info("Compute time_varying_elastance")
                    d = utils.compute_time_varying_elastance(patient, params, val)
                    self._save_tmp_results(patient_name, "time_varying_elastance", d)
                    data[patient_name].update(**d)

            if "emax" in args:

                if self._data_exist(patient_name, "emax") and not self._recompute:
                    self._load_tmp_results(patient_name, "emax", data[patient_name])

                else:
                    logger.info("Compute emax")

                    d = {
                        "emax": utils.compute_emax(
                            patient, params, val, self._valve_times[patient_name]
                        )
                    }

                    self._save_tmp_results(patient_name, "emax", d)
                    data[patient_name].update(**d)

            if "end_systolic_elastance" in args:

                if (
                    self._data_exist(patient_name, "end_systolic_elastance")
                    and not self._recompute
                ):
                    self._load_tmp_results(
                        patient_name, "end_systolic_elastance", data[patient_name]
                    )

                else:
                    logger.info("Compute end_systolic_elastance")

                    es = self._es[patient_name]

                    state = val["states"][str(es)]
                    gamma = val["gammas"][str(es)]

                    matparams = val["material_parameters"]

                    if params["matparams_space"] == "regional":
                        sfun = merge_control(patient, params["merge_passive_control"])
                        mat = RegionalParameter(sfun)
                        mat.vector()[:] = matparams["a"]
                    else:
                        family, degree = params["matparams_space"].split("_")
                        mat_space = dolfin.FunctionSpace(
                            moving_mesh, family, int(degree)
                        )
                        mat = dolfin.Function(mat_space, name="material_parameter_a")
                        mat.vector()[:] = matparams["a"]

                    matparams["a"] = mat

                    for k in ["a_f", "b", "b_f"]:
                        v = dolfin.Constant(matparams[k][0])
                        matparams[k] = v

                    d = {}
                    if patient.is_biv():
                        pressure = (patient.pressure[es], patient.rv_pressure[es])

                        d["end_systolic_elastance_rv"] = utils.compute_elastance(
                            state,
                            pressure,
                            gamma,
                            patient,
                            params,
                            matparams,
                            chamber="rv",
                        )

                    else:
                        pressure = patient.pressure[es]

                    d["end_systolic_elastance"] = utils.compute_elastance(
                        state, pressure, gamma, patient, params, matparams, chamber="lv"
                    )

                    self._save_tmp_results(patient_name, "end_systolic_elastance", d)
                    data[patient_name].update(**d)

            if "gamma_mean" or "gamma_regional" in args:

                dX = dolfin.Measure(
                    "dx", subdomain_data=patient.sfun, domain=patient.mesh
                )
                regions = [int(r) for r in set(patient.sfun.array())]

                gs = val["gammas"]
                gamma_lst = [gs[k] for k in sorted(gs, key=utils.asint)]

                if params["gamma_space"] == "regional":
                    sfun = merge_control(patient, params["merge_active_control"])
                    gamma = RegionalParameter(sfun)
                else:
                    gamma_space = dolfin.FunctionSpace(patient.mesh, "CG", 1)
                    gamma = dolfin.Function(gamma_space, name="Contraction Parameter")

                if "gamma_mean" in args:
                    if (
                        self._data_exist(patient_name, "gamma_mean")
                        and not self._recompute
                    ):
                        self._load_tmp_results(
                            patient_name, "gamma_mean", data[patient_name]
                        )

                    else:
                        logger.info("Compute mean gamma")
                        d = {
                            "gamma_mean": utils.get_global(
                                dX, gamma, gamma_lst, regions, params["T_ref"]
                            )
                        }
                        self._save_tmp_results(patient_name, "gamma_mean", d)
                        data[patient_name].update(**d)

                if "gamma_regional" in args:
                    if (
                        self._data_exist(patient_name, "gamma_regional")
                        and not self._recompute
                    ):
                        self._load_tmp_results(
                            patient_name, "gamma_regional", data[patient_name]
                        )

                    else:
                        logger.info("Compute regional gamma")
                        d = {
                            "gamma_regional": utils.get_regional(
                                dX, gamma, gamma_lst, regions, params["T_ref"]
                            )
                        }
                        self._save_tmp_results(patient_name, "gamma_regional", d)
                        data[patient_name].update(**d)

            if "geometric_distance" in args:

                if self._data_exist(patient_name, "geometric_distance"):
                    self._load_tmp_results(
                        patient_name, "geometric_distance", data[patient_name]
                    )
                else:
                    logger.info("Compute geometric distance")
                    vtk_output = "/".join(
                        [self._outdir, "surface_files2", patient_name]
                    )
                    d = utils.compute_geometric_distance(
                        patient, val["displacements"], vtk_output
                    )
                    self._save_tmp_results(patient_name, "geometric_distance", d)
                    data[patient_name].update(**d)

            if "data_mismatch" in args:

                if self._data_exist(patient_name, "data_mismatch"):
                    self._load_tmp_results(
                        patient_name, "data_mismatch", data[patient_name]
                    )
                else:
                    logger.info("Compute data mismatch")
                    d = utils.copmute_data_mismatch(
                        val["displacements"],
                        patient["geometry"],
                        patient["data"]["volume"],
                        patient["data"]["strain"],
                    )

                    self._save_tmp_results(patient_name, "data_mismatch", d)
                    data[patient_name].update(**d)

            if "vtk_simulation" in args or "mechanical_features" in args:

                output = "/".join(
                    [os.path.dirname(self._geoname), "features", patient_name + ".h5"]
                )

                if not os.path.isfile(output) or self._recompute:
                    logger.info("Compute meachanical features")
                    d = {
                        "features_scalar": utils.copmute_mechanical_features(
                            patient, params, val, output, keys=self._feature_keys
                        )
                    }
                    self._save_tmp_results(patient_name, "features_scalar", d)

                self._load_tmp_results(
                    patient_name, "features_scalar", data[patient_name]
                )
                features = {patient_name: load.load_dict_from_h5(output)}

                self._update_features(features)

            if "mechanical_features_scalar" in args:

                if not os.path.isfile(output) or self._recompute:
                    logger.info("Compute meachanical features")
                    d = {
                        "features_scalar": utils.copmute_mechanical_features(
                            patient, params, val, output, keys=self._feature_keys
                        )
                    }
                    self._save_tmp_results(patient_name, "features_scalar", d)

                self._load_tmp_results(
                    patient_name, "features_scalar", data[patient_name]
                )

            if any([a.startswith("cardiac_work") for a in args]):

                try:
                    idx = np.where([a.startswith("cardiac_work") for a in args])[0][0]
                    string = args[idx]
                    _, case, wp = string.split(":")

                except:
                    string = "cardiac_work:comp_long:SE"
                    _, case, wp = string.split(":")

                assert (
                    wp in work_pairs
                ), "Illegal work pair: {}. Legal inputs:{}".format(wp, work_pairs)
                assert case in cases, "Illegal case: {}. Legal inputs:{}".format(
                    case, cases
                )

                if self._data_exist(patient_name, string) and not self._recompute:
                    self._load_tmp_results(patient_name, string, data[patient_name])

                else:

                    if 0:  # string == 'cardiac_work:comp_long:pgradu':

                        strain = utils.get_regional_strains(
                            val["displacements"], patient, **params
                        )
                        work = {}
                        pressure = (
                            patient.pressure[1:]
                            if params["unload"]
                            else patient.pressure
                        )
                        for r, s_ in strain["longitudinal"].items():

                            s = s_[1:] if params["unload"] else s_
                            work[r] = utils.compute_cardiac_work_echo(
                                pressure, s, flip=True
                            )

                        data[patient_name]["work_pgradu_comp_long"] = work
                    else:

                        logger.info("Compute cardiac work")
                        d = utils.compute_cardiac_work(patient, params, val, case, wp)

                        self._save_tmp_results(patient_name, string, d)
                        data[patient_name].update(**d)

                data[patient_name]["measured_work"] = compute_echo_work(patient, params)
                # data[patient_name]["measured_work"] \
                # ={k:np.transpose(v)[2].tolist() for k, v in patient.work.iteritems()}

        self._update_results(data)

    def print_timings(self):

        timings = {k: [] for k in list(self._data[self._keys[0]]["timings"].keys())}
        ncontrols = []

        for patient_name in self._keys:

            ncontrols.append(len(self._data[patient_name]["gammas"]["0"]))
            for k, v in self._data[patient_name]["timings"].items():

                timings[k].append(v)

        forward_times = np.concatenate(
            [f.tolist() for t in timings["forward_times"] for f in list(t.values())]
        )

        print(
            (
                "\nForward times = {} +/- {}".format(
                    np.mean(forward_times), np.std(forward_times)
                )
            )
        )

        backward_times = np.concatenate(
            [f.tolist() for t in timings["backward_times"] for f in list(t.values())]
        )

        print(
            (
                "\nBackward times = {} +/- {}".format(
                    np.mean(backward_times), np.std(backward_times)
                )
            )
        )

        print(
            (
                "\nNumber of controls = {} +/- {}".format(
                    np.mean(ncontrols), np.std(ncontrols)
                )
            )
        )

        run_times = np.concatenate([f.tolist() for f in timings["run_time"]])

        print(("\nRun time = {} +/- {}".format(np.mean(run_times), np.std(run_times))))

    def make_simulation(self, refined=False):

        setup_general_parameters()

        # Make sure that we have computed the features
        keys = list(self._features.keys())
        if len(keys) == 0:
            self.compute("vtk_simulation")

        logger.info("#" * 40 + "\nMake simulation")

        main_outdir = "/".join([self._outdir, "simulation"])
        if not os.path.exists(main_outdir):
            os.makedirs(main_outdir)

        for patient_name in list(self._results.keys()):

            logger.info("\tPatient: {}".format(patient_name))
            outdir = "/".join([main_outdir, patient_name])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            params, patient = self._setup_compute(patient_name)
            # params = self._params

            # params["Patient_parameters"]["patient"] = patient_name
            # params["Patient_parameters"]["mesh_path"] \
            #     = self._mesh_path.format(patient_name)
            # params["Patient_parameters"]["pressure_path"] \
            #     = self._pressure_path.format(patient_name)

            # utils.save_displacements(params, self._features[patient_name], outdir)
            if refined:
                utils.make_refined_simulation(
                    params,
                    self._features[patient_name],
                    outdir,
                    patient,
                    self._data[patient_name],
                )
            else:
                utils.make_simulation(
                    params,
                    self._features[patient_name],
                    outdir,
                    patient,
                    self._data[patient_name],
                )

        logger.info("#" * 40)

    def set_feature_keys(self, *args):
        self._feature_keys = args
