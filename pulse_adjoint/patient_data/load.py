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
import yaml, os

try:
    import h5py
except:
    pass
try:
    from pulse.geometry_utils import load_geometry_from_h5
except:
    msg = (
        "\n\nWARNING. TOOLS FOR LOADING MESH IS MISSING\n"
        "Get repo: git clone git@bitbucket.org:finsberg/mesh_generation.git\n"
    )
    raise ImportError(msg)

from .utils import *

STRAIN_REGIONS = {
    "LVBasalAnterior": 1,
    "LVBasalAnteroseptal": 2,
    "LVBasalSeptum": 3,
    "LVBasalInferior": 4,
    "LVBasalPosterior": 5,
    "LVBasalLateral": 6,
    "LVMidAnterior": 7,
    "LVMidAnteroseptal": 8,
    "LVMidSeptum": 9,
    "LVMidInferior": 10,
    "LVMidPosterior": 11,
    "LVMidLateral": 12,
    "LVApicalAnterior": 13,
    "LVApicalSeptum": 14,
    "LVApicalInferior": 15,
    "LVApicalLateral": 16,
    "LVApex": 17,
}

STRAIN_DIRECTIONS = [
    "RadialStrain",
    "LongitudinalStrain",
    "CircumferentialStrain",
    "AreaStrain",
]


def read_volume_data(filename):
    """Get volume data from echo file.
        
    *Arguments*
      filename (sting)
        path to file containing echo data
    
    *Returns*
      (volume in ml, times)
    """
    with h5py.File(filename, "r") as h5file:
        volume = np.asarray(h5file["LV_Volume_Trace"])
        times = np.asarray(h5file["time_stamps"])

    volume_ml = m3_2_ml(volume.tolist())

    return volume_ml, times


def load_geometry(h5name, h5group, **kwargs):
    """FIXME! briefly describe function

    :param h5name: 
    :param h5group: 
    :returns: 
    :rtype: 

    """

    # Check if there is provided any fiber angles.
    # If not, use the default ones
    fiber_angle_epi = kwargs.pop("fiber_angle_epi", -60)
    fiber_angle_endo = kwargs.pop("fiber_angle_endo", 60)
    sheet_angle_epi = kwargs.pop("sheet_angle_epi", 0)
    sheet_angle_endo = kwargs.pop("sheet_angle_endo", 0)
    include_sheets = kwargs.pop("include_sheets", False)

    geo = load_geometry_from_h5(
        h5name, h5group, fiber_angle_endo, fiber_angle_epi, include_sheets
    )

    return geo


def get_3d_strain(echo_data, ref_time=0):
    """Get strain vaules for given reference time
    Note: This does not give Global strain nor Area strain
    
    *Arguments*
    ref_time (int)
    Index from where you want you list to start
    
    *Returns*
    A dictionary with all strain values for circumferential
    longitudinal and radial direction. out[1] gives a list of
    tripples (circumferential, longitudinal, radial) for region
    1 (LVBasalAnterior) starting with zero at reference time (ref_time)
    """
    out = {}
    original_strain = {}

    with h5py.File(echo_data, "r") as echo_file:
        hdf = echo_file["/LV_Strain_Trace"]
        traces = h5py2dict(hdf)
        times = np.array(echo_file["time_stamps"])

    for region in STRAIN_REGIONS:
        strain = []
        strain_orig = []
        for direction in [
            "CircumferentialStrain",
            "RadialStrain",
            "LongitudinalStrain",
        ]:
            key = "_".join([direction, region])
            if key in traces:

                trace_orig = traces[key]
                trace_corrected = correct_drift(trace_orig, use_spline=True)
                trace = calibrate_strain(
                    trace_corrected, ref_time, relative_strain=True
                )

            else:
                trace_orig = trace = np.zeros(len(times))

                msg = (
                    "Warning: Strain measurement does not exists for "
                    "\{}. Force zero strain in this region".format(key)
                )
                logger.info(msg)

            strain.append(trace.tolist())
            strain_orig.append(trace_orig.tolist())

        out[STRAIN_REGIONS[region]] = np.transpose(strain).tolist()
        original_strain[STRAIN_REGIONS[region]] = np.transpose(strain_orig).tolist()

    return out, original_strain


def get_echo_surfaces(echo_path):

    data = {}
    with h5py.File(echo_path, "r") as echo_file:

        epi = echo_file["/LV_Mass_Epi"]
        endo = echo_file["/LV_Mass_Endo"]
        strain_mesh = echo_file["/LV_Strain/mesh"]

        data["epi"] = h5py2dict(epi)
        data["endo"] = h5py2dict(endo)
        data["strain_mesh"] = h5py2dict(strain_mesh)

    return data


def get_traces_from_excel(echo_data, pressure_data):

    import pandas as pd

    xl_pressure_data = os.path.splitext(pressure_data)[0] + ".xlsx"
    if not os.path.isfile(xl_pressure_data):
        raise IOError("File {} does not exist".format(xl_pressure_data))

    traces = Object()

    with pd.ExcelFile(xl_pressure_data) as xl:

        traces.general = xl.parse("General")
        traces.pressure_traces = xl.parse("PressureTraces")
        traces.work_traces = xl.parse("Work Traces")
        traces.segments = xl.parse("Segments")
        traces.strain_traces = xl.parse("Strain Traces")

    return traces


def load_measurement(pressure_path, echo_path, **kwargs):

    d = {}
    if pressure_path != "":
        with open(pressure_path, "rb") as output:
            d = yaml.load(output)
    else:
        logger.warning("Pressure path does not exist")

    if echo_path != "":

        strain, original_strain = get_3d_strain(echo_path, d["passive_filling_begins"])

        if "strain" in d:

            d["strain_3d"] = strain
            d["original_strain_3d"] = original_strain

        else:
            d["strain"] = strain
            d["original_strain"] = original_strain

    else:
        logger.warning("Echo path does not exist")

    return d
