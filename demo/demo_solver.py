"""
The solver implemented in Pulse-Adjoint
works as a stand-alone solver for 
cardiac mechanics.

This script shows how to use this solver using 
your own geometry.
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
import numpy as np
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.material import HolzapfelOgden, NeoHookean
from pulse_adjoint.setup_optimization import  setup_solver_parameters, setup_general_parameters
from pulse_adjoint.adjoint_contraction_args import logger
from pulse_adjoint.utils import QuadratureSpace
import os

logger.setLevel(DEBUG)
path = os.path.dirname(os.path.abspath(__file__))

def load_patient_data():
    import yaml
    import numpy as np
    
    class Object: pass
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

        # Get fibers
        V = QuadratureSpace(mesh, 4)
        name = "fiber"
        l = Function(V, name = name)
        fsubgroup = fgroup+"/fiber_epi-60_endo60"
        h5file.read(l, fsubgroup)
        fsub_attrs = h5file.attributes(fsubgroup)
        setattr(patient, "e_f", l)
        
    # You don't need sheets nor cross-sheets
    setattr(patient, "e_s", None)
    setattr(patient, "e_sn", None)
        
    # Set some markers
    setattr(patient, 'ENDO',  30)
    setattr(patient, 'BASE', 10)

    return patient

    
def demo_heart():

    setup_general_parameters()
    patient = load_patient_data()
    
    
    mesh = patient.mesh
    ffun = patient.facets_markers
    N = FacetNormal(mesh)


    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        # no_base_x_tran_bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), patient.BASE)
        no_base_x_tran_bc = DirichletBC(V.sub(0), Constant(0.0), patient.BASE)
        return no_base_x_tran_bc


    # Fibers
    f0 = patient.e_f
  
    # Contraction parameter
    # gamma = Constant(0.0)
    gamma = Function(FunctionSpace(mesh, "R", 0))
    # Pressure
    pressure = Expression("t", t = 0.0)

    # Spring
    spring = Constant(0.1)

    
    # Set up material model
    matparams = {"a":2.28, "a_f":1.685, 
                "b":9.726, "b_f":15.779}
    material = HolzapfelOgden(patient.e_f, gamma, matparams, active_model = "active_strain")
    # material = HolzapfelOgden(f0, gamma, active_model = "active_stress")
    # material = NeoHookean(f0, gamma, active_model = "active_stress")

    # Solver parameters
    solver_parameters = {"snes_solver":{}}
    solver_parameters["nonlinear_solver"] = "snes"
    solver_parameters["snes_solver"]["method"] = "newtontr"
    solver_parameters["snes_solver"]["maximum_iterations"] = 50
    solver_parameters["snes_solver"]["absolute_tolerance"] = 1e-5
    solver_parameters["snes_solver"]["linear_solver"] = "lu"

    # solver_parameters = {"newton_solver":{}}
    # solver_parameters["nonlinear_solver"] = "newton"
    # solver_parameters["newton_solver"]["method"] = "newtontr"
    # solver_parameters["newton_solver"]["maximum_iterations"] = 8
    # solver_parameters["newton_solver"]["absolute_tolerance"] = 1e-8
    # solver_parameters["ewton_solver"]["linear_solver"] = "lu"

    # Create parameters for the solver
    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, patient.ENDO]],
                   "robin":[[spring, patient.BASE]]},
             "solve":solver_parameters}

    parameters["adjoint"]["stop_annotating"] = True

    # Initialize solver
    solver = LVSolver(params)

    # Solve for the initial state
    solver.solve()
    u,p = solver.get_state().split()
    # u = solver.get_state()#.split()
    plot(u, mode="displacement", title = "Initial solve")

    # Put on some pressure and solve
    pressure.t = 0.1
    solver.solve()
        
    u,p = solver.get_state().split()
    # u = solver.get_state()#.split()
    plot(u, mode="displacement",
         title = "Soulution after pressure change")
    # plot(p, title = "hydrostatic pressure")

    # Put on some active contraction and solve
    gamma.assign(Constant(0.05))
    solver.solve()
        
    u,p = solver.get_state().split()
    # u = solver.get_state()#.split()
    plot(u, mode="displacement",
         title = "Solution after initiation of active contraction")
    # plot(p, title = "hydrostatic pressure")
    
    interactive()

def demo_cube():
    setup_general_parameters()
    mesh = UnitCubeMesh(3,3,3)

    # Make some simple boundary conditions
    class Right(SubDomain):
        def inside(self, x, on_boundary): 
            return x[0] > (1.0 - DOLFIN_EPS) and on_boundary
    class Left(SubDomain):
        def inside(self, x, on_boundary): 
            return x[0] < DOLFIN_EPS and on_boundary
    class TopBottom(SubDomain):
        def inside(self, x, on_boundary):
            return (x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS) and on_boundary

    
    # Mark boundaries
    ffun = MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    left = Left()
    left_marker = 1
    left.mark(ffun, left_marker)

    right = Right()
    right_marker = 2
    right.mark(ffun, right_marker)

    topbottom = TopBottom()
    topbottom_marker = 3
    topbottom.mark(ffun, topbottom_marker)

    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        no_base_x_tran_bc = DirichletBC(V.sub(0), 0, topbottom_marker)
        return no_base_x_tran_bc

    # Spring Constant for Robin Condition
    spring = Constant(0.1, name ="spring_constant")

    # Facet Normal
    N = FacetNormal(mesh)

    # Pressure
    pressure = Expression("t", t = 0.1)

    # Fibers
    V_f = QuadratureSpace(mesh, 4)
    # V_f = VectorFunctionSpace(mesh, "CG", 1)
    # Unit field in x-direction
    f0 = interpolate(Expression(("1.0", "0.0", "0.0")), V_f)

    # Contraction parameter
    gamma = Constant(0.1)
    
    # Set up material model
    material = HolzapfelOgden(f0, gamma, active_model = "active_stress")
    
    # Solver parameters
    solver_parameters = setup_solver_parameters()
    solver_parameters = {"snes_solver":{}}
    solver_parameters["nonlinear_solver"] = "snes"
    solver_parameters["snes_solver"]["method"] = "newtonls"
    solver_parameters["snes_solver"]["maximum_iterations"] = 8
    solver_parameters["snes_solver"]["absolute_tolerance"] = 1e-5
    solver_parameters["snes_solver"]["linear_solver"] = "lu"

    # Create parameters for the solver
    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             "state_space": "P_2:P_1",
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, left_marker]],
                   "robin":[[spring, right_marker]]},
             "solve":solver_parameters}

    solver = LVSolver(params)
    
    solver.solve()

    u,p = solver.get_state().split()
    plot(u, mode="displacement", title = "displacement")
    plot(p, title = "hydrostatic pressure")

    postprocess = solver.postprocess()

    fiber_stress = postprocess.fiber_stress()
    plot(fiber_stress, title = "fiber stress")

    
    interactive()

if __name__ == "__main__":
    # demo_cube()
    demo_heart()

