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
from campass.lvsolver import LVSolver
from campass.material import HolzapfelOgden
from campass.setup_optimization import setup_application_parameters, setup_solver_parameters, setup_general_parameters
from campass.adjoint_contraction_args import logger

logger.setLevel(DEBUG)

def test_solver():
    setup_general_parameters()
    params = setup_application_parameters()
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

    # Spring Constant for Robin Condition
    spring = Constant(0.1, name ="spring_constant")

    # Facet Normal
    N = FacetNormal(mesh)

    # Pressure
    pressure = Expression("t", t = 0.1)

    # Fibers
    V_f = VectorFunctionSpace(mesh, "Quadrature", 4)
    # Unit field in x-direction
    f0 = interpolate(Expression(("1.0", "0.0", "0.0")), V_f)
    
    # Set up material model
    material = HolzapfelOgden(f0, active_model = "active_strain")
    

    # Create parameters for the solver
    solver_parameters = {"mesh": mesh,
                         "facet_function": ffun,
                         "facet_normal": N,
                         "state_space": "P_2:P_1",
                         "compressibility":{"type": "incompressible",
                                            "lambda":0.0},
                         "material": material,
                         "bc":{"dirichlet": make_dirichlet_bcs,
                               "neumann":[[pressure, left_marker]],
                               "robin":[[spring, right_marker]]},
                         "solve":setup_solver_parameters()}

    solver = LVSolver(solver_parameters)
    
    solver.solve()

    u,p = solver.get_state().split()
    plot(u, mode="displacement")
    interactive()

if __name__ == "__main__":
    test_solver()
