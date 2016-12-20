"""
This is a demo for how to use Pulse-Adjoint yourselves
without the additional packages "mesh_generation"
and "patient". 

This script shows how to load your patient data,
and use this as input to the optimization.

The patient data used for this demo is found in
the folder "pulse_adjoint/demo/data" withing th
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
from dolfin import *
from dolfin_adjoint import *
from pulse_adjoint.run_optimization import run_passive_optimization, run_active_optimization
from pulse_adjoint.adjoint_contraction_args import *
from pulse_adjoint.utils import passive_inflation_exists, Text, pformat, Object
from pulse_adjoint.setup_optimization import setup_adjoint_contraction_parameters, setup_general_parameters

path = os.path.dirname(os.path.abspath(__file__))


def load_patient_data():
    import yaml
    import numpy as np
    
    
    patient = Object()

    h5group = "22"
    ggroup = '{}/geometry'.format(h5group)
    mgroup = '{}/mesh'.format(ggroup)
    lgroup = "{}/local basis functions".format(h5group)
    fgroup = "{}/microstructure/".format(h5group)

    h5name = "/".join([path, "data/mesh.h5"])

    with HDF5File(mpi_comm_world(), h5name, "r") as h5file:

        # Load mesh
        mesh = Mesh(mpi_comm_world())
        h5file.read(mesh, mgroup, False)
        patient.mesh = mesh

        # Get facet function
        ffun = MeshFunction("size_t", mesh, 2, mesh.domains())
        ffun.array()[ffun.array() == max(ffun.array())] = 0
        patient.facets_markers = ffun

        # Get cell function
        sfun = MeshFunction("size_t", mesh, 3, mesh.domains())
        patient.strain_markers = sfun

        # Get local bais functions
        local_basis_attrs = h5file.attributes(lgroup)
        lspace = local_basis_attrs["space"]
        family, order = lspace.split('_')

        namesstr = local_basis_attrs["names"]
        names = namesstr.split(":")
        
        lb_names = {"circumferential":"e_circ", "radial":"e_rad", "longitudinal":"e_long"}
        V = VectorFunctionSpace(mesh, family, int(order))
        for name in names:
            l = Function(V, name = name)
            h5file.read(l, lgroup+"/{}".format(name))
            setattr(patient, lb_names[name], l)

        # Get fibers
        if DOLFIN_VERSION_MAJOR > 1.6:
            elm = VectorElement(family = "Quadrature",
                                cell = mesh.ufl_cell(),
                                degree = 4,
                                quad_scheme="default")
            V = FunctionSpace(mesh, elm)
        else:
            V = VectorFunctionSpace(mesh, "Quadrature", 4)
            
        name = "fiber"
        l = Function(V, name = name)
        fsubgroup = fgroup+"/fiber_epi-60_endo60"
        h5file.read(l, fsubgroup)
        fsub_attrs = h5file.attributes(fsubgroup)
        setattr(patient, "e_f", l)
        
    # You don't need sheets nor cross-sheets
    setattr(patient, "e_s", None)
    setattr(patient, "e_sn", None)
    setattr(patient, "mesh_type", lambda: "lv")
        
    # Set some markers
    setattr(patient, 'ENDO',  30)
    setattr(patient, 'BASE', 10)

    # Set equal weights on the strain regions in the optmization
    patient.strain_weights = np.ones((17, 3))

    
    measurements = "/".join([path, "data/measurements.yml"])
    with open(measurements, "rb" ) as data:
        d = yaml.load(data)
        
    patient.pressure = np.array(d["pressure"])
    patient.volume = np.array(d["volume"])
    patient.strain = d["strain"]
    patient.original_strain = d["original_strain"]

    
    patient.passive_filling_duration = 3
    patient.num_contract_points = 31
    patient.num_points = 34

    patient.markers = {"BASE": (10, 2), "ENDO":(30, 2), "EPI": (40, 2)}

    return patient

def main(params):

    setup_general_parameters()
    logger.info(Text.blue("Start Adjoint Contraction"))
    logger.info(pformat(params.to_dict()))
    

    ############# GET PATIENT DATA ##################
    patient = load_patient_data()

    ############# RUN MATPARAMS OPTIMIZATION ##################
    
    # Make sure that we choose passive inflation phase
    params["phase"] =  PHASES[0]
    if not passive_inflation_exists(params):
        run_passive_optimization(params, patient)
        adj_reset()
    
    ################## RUN GAMMA OPTIMIZATION ###################

    # Make sure that we choose active contraction phase
    params["phase"] = PHASES[1]
    run_active_optimization(params, patient)


if __name__ == '__main__':
    
    params = setup_adjoint_contraction_parameters()
    main(params)
    
    
