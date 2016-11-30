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
from pulse_adjoint.utils import QuadratureSpace
from pulse_adjoint.setup_optimization import setup_application_parameters, setup_solver_parameters, setup_general_parameters
from pulse_adjoint.adjoint_contraction_args import logger
logger.setLevel(DEBUG)


def pressure_increase_passive():
    """
    Make sure that exceptions are handled correctly
    it the solver fails to increase the pressure in 
    the passive phase
    """
    
    from utils import setup_params as setup_test_params
    from pulse_adjoint.utils import Text, pformat, UnableToChangePressureExeption
    from pulse_adjoint.setup_optimization import initialize_patient_data, setup_simulation
    from pulse_adjoint.run_optimization import run_passive_optimization_step
    
    
    params = setup_test_params()
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])

    
    patient.pressure = np.array([  0,  2e5,  4e5])
    v = patient.volume[0]
    # from IPython import embed; embed()
    # exit()
    patient.volume = np.array([  v,  2*v,  4*v])
    logger.info(Text.blue("\nTest Pressure Increase"))

    logger.info(pformat(params.to_dict()))


    params["phase"] = "passive_inflation"
    # params["alpha_matparams"] = 0.5
    
      
    sucess = False
    point = 0
    while not sucess:
        
        try:
            measurements, solver_parameters, p_lv, paramvec = \
              setup_simulation(params, patient)
            rd, paramvec = run_passive_optimization_step(params, 
                                                        patient, 
                                                        solver_parameters, 
                                                        measurements, 
                                                        p_lv, paramvec)
        except UnableToChangePressureExeption as ex:
            logger.info("Unable to increase pressure. Exception caught")
            
            logger.info("\n Data before interpolation")
            logger.info("Pressure = {}".format(patient.pressure))
            logger.info("Volume = {}".format(patient.volume))
            patient.interpolate_data(point)
            logger.info("\n Data after interpolation")
            logger.info("Pressure = {}".format(patient.pressure))
            logger.info("Volume = {}".format(patient.volume))
            adj_reset()
        else:
            print "Success"
            sucess = True

def pressure_increase_active():
    """
    Make sure that exceptions are handled correctly
    it the solver fails to increase the pressure in 
    the active phase
    """
    
    from utils import setup_params as setup_test_params
    from pulse_adjoint.utils import Text, pformat, UnableToChangePressureExeption, passive_inflation_exists
    from pulse_adjoint.setup_optimization import initialize_patient_data, setup_simulation
    from pulse_adjoint.run_optimization import run_passive_optimization, run_active_optimization_step
    
    
    params = setup_test_params()
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])

    
    patient.pressure = np.array([  0,  2,  4, 6e5])
    v = patient.volume[0]
    # from IPython import embed; embed()
    # exit()
    patient.volume = np.array([  v,  2*v,  4*v, 6*v])
    logger.info(Text.blue("\nTest Pressure Increase"))

    logger.info(pformat(params.to_dict()))


    if not passive_inflation_exists(params):
        params["phase"] = "passive_inflation"
        params["optimize_matparams"] = False
        run_passive_optimization(params, patient)
   
    
      
    sucess = False
    point = 0
    while not sucess:
        
        try:
            measurements, solver_parameters, p_lv, paramvec = \
              setup_simulation(params, patient)
            rd, paramvec = run_passive_optimization_step(params, 
                                                        patient, 
                                                        solver_parameters, 
                                                        measurements, 
                                                        p_lv, paramvec)
        except UnableToChangePressureExeption as ex:
            logger.info("Unable to increase pressure. Exception caught")
            
            logger.info("\n Data before interpolation")
            logger.info("Pressure = {}".format(patient.pressure))
            logger.info("Volume = {}".format(patient.volume))
            patient.interpolate_data(point)
            logger.info("\n Data after interpolation")
            logger.info("Pressure = {}".format(patient.pressure))
            logger.info("Volume = {}".format(patient.volume))
            adj_reset()
        else:
            print "Success"
            sucess = True
        
        
def test_solver_heart():

    setup_general_parameters()
    params = setup_application_parameters()

    from patient_data import TestPatient

    patient = TestPatient()
    

    mesh = patient.mesh
    ffun = patient.facets_markers
    N = FacetNormal(mesh)
    # element_type = "mini"
    element_type = "taylor_hood"


    # Dirichlet BC
    def make_dirichlet_bcs(W):
        V = W.sub(0)
        if element_type == "mini":
            P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
            B  = VectorFunctionSpace(mesh, "Bubble", 4)
            V1 = P1+B
            zero = project(Constant((0, 0, 0)), V1)
        else:
            zero = Constant((0,0,0))
            
        no_base_x_tran_bc = DirichletBC(V, zero, patient.BASE)
        
        # V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        # no_base_x_tran_bc = DirichletBC(V.sub(0), 0, patient.BASE)
        return no_base_x_tran_bc


    # Fibers
    V_f = QuadratureSpace(mesh, 4)
    # Unit field in x-direction
    f0 = patient.e_f
    
    # from IPython import embed; embed()
    # exit()

    # Contraction parameter
    gamma = Constant(0.0)

    # Pressure
    pressure = Expression("t", t = 0)

    # Spring
    spring = Constant(0.0)

    matparams = {"a":1.0, "a_f":1.0, 
                 "b":5.0, "b_f":5.0}
    # Set up material model
    # material = HolzapfelOgden(f0, gamma, active_model = "active_stress")
    material = HolzapfelOgden(f0, gamma, matparams, active_model = "active_strain")
    # material = NeoHookean(f0, gamma, active_model = "active_stress")

    # Solver parameters
    solver_parameters = {"snes_solver":{}}
    solver_parameters["nonlinear_solver"] = "snes"
    solver_parameters["snes_solver"]["method"] = "newtontr"
    solver_parameters["snes_solver"]["maximum_iterations"] = 8
    solver_parameters["snes_solver"]["absolute_tolerance"] = 1e-5
    solver_parameters["snes_solver"]["linear_solver"] = "lu"

    # solver_parameters = {"newton_solver":{}}
    # solver_parameters["nonlinear_solver"] = "newton"
    # solver_parameters["newton_solver"]["method"] = "newtontr"
    # solver_parameters["newton_solver"]["maximum_iterations"] = 8
    # solver_parameters["newton_solver"]["absolute_tolerance"] = 1e-8
    # solver_parameters["newton_solver"]["linear_solver"] = "lu"

    # Create parameters for the solver
    params= {"mesh": mesh,
             "facet_function": ffun,
             "facet_normal": N,
             # "state_space": "P_2:P_1",
             "elements": element_type, 
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, patient.ENDO]],
                   "robin":[[spring, patient.BASE]]},
             "solve":solver_parameters}

    parameters["adjoint"]["stop_annotating"] = True
    solver = LVSolver(params)

    solver.solve()
    # u,p = solver.get_state().split()
    # # u = solver.get_state().split()
    # plot(u, mode="displacement", title = "displacement")
    # plot(p, title = "hydrostatic pressure")

    # postprocess = solver.postprocess()

    # fiber_stress = postprocess.fiber_stress()
    # plot(fiber_stress, title = "fiber stress")
    
    # interactive()

def test_solver_cube():
    setup_general_parameters()
    params = setup_application_parameters()
    mesh = UnitCubeMesh(3,3,3)

    # element_type = "taylor_hood"
    element_type = "mini"
    
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
        V = W.sub(0)
        if element_type == "mini":
            P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
            B  = VectorFunctionSpace(mesh, "Bubble", 4)
            V1 = P1+B
            zero = project(Constant((0, 0, 0)), V1)
        else:
            zero = Constant((0,0,0))
            
        no_base_x_tran_bc = DirichletBC(V, zero, topbottom_marker)
        return no_base_x_tran_bc

    # Spring Constant for Robin Condition
    spring = Constant(0.0, name ="spring_constant")

    # Facet Normal
    N = FacetNormal(mesh)

    # Pressure
    pressure = Expression("t", t = 0.1)

    # Fibers
    V_f =QuadratureSpace(mesh, 4)
    # V_f = VectorFunctionSpace(mesh, "CG", 1)
    # Unit field in x-direction
    f0 = interpolate(Expression(("1.0", "0.0", "0.0")), V_f)

    # Contraction parameter
    gamma = Constant(0.1)
    
    # Set up material model
    material = HolzapfelOgden(f0, gamma, active_model = "active_stress")
    
    # Solver parameters
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
             "elements": element_type, #"mini", #"taylor_hood", 
             "compressibility":{"type": "incompressible",
                                "lambda":0.0},
             "material": material,
             "bc":{"dirichlet": make_dirichlet_bcs,
                   "neumann":[[pressure, left_marker]],
                   "robin":[[spring, right_marker]]},
             "solve":solver_parameters}

    solver = LVSolver(params)
    
    solver.solve()

    # u,p = solver.get_state().split()
    # plot(u, mode="displacement", title = "displacement")
    # plot(p, title = "hydrostatic pressure")

    # postprocess = solver.postprocess()

    # V_cg = FunctionSpace(mesh, "CG", 1)
    # V_quad = QuadratureSpace(mesh, 4)


    # fiber_stress = postprocess.fiber_stress()
    # plot(fiber_stress, title = "fiber stress")

    # interactive()

if __name__ == "__main__":
    # test_solver_cube()
    test_solver_heart()
    # test_pressure_increase_passive()
    # test_pressure_increase_active()
