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
import os, logging
import dolfin


log_level = logging.INFO

# Setup logger
def make_logger(name, level = logging.INFO):
    import logging

    mpi_filt = lambda: None
    def log_if_proc0(record):
        if dolfin.mpi_comm_world().rank == 0:
            return 1
        else:
            return 0
    mpi_filt.filter = log_if_proc0

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    ch = logging.StreamHandler()
    ch.setLevel(0)
    formatter = logging.Formatter('%(message)s') #'\n%(name)s - %(levelname)s - %(message)s\n'
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addFilter(mpi_filt)

   
    
    
    dolfin.set_log_active(False)
    dolfin.set_log_level(dolfin.WARNING)

    # ffc_logger = logging.getLogger('FFC')
    # ffc_logger.setLevel(DEBUG)
    # ffc_logger.addFilter(mpi_filt)
    

    # ufl_logger = logging.getLogger('UFL')
    # ufl_logger.setLevel(DEBUG)
    # ufl_logger.addFilter(mpi_filt)
    
    # from haosolver import logger as hao_logger
    # hao_logger.setLevel(DEBUG)

    return logger

logger = make_logger("Adjoint_Contraction", log_level)


############### OPTIMIZATION PARAMETERS ######################
# Strain weights for optimization
# What kind of weighting rule 
WEIGHT_RULES = ["equal", "drift", "peak_value", "combination"]
DEFAULT_WEIGHT_RULE = "equal" 
# Any preferred direction for weighting ("c", "l", "r" or None)
WEIGHT_DIRECTIONS = ["c", "l", "r", "all"]
DEFAULT_WEIGHT_DIRECTION = "all"

# The different phases we can optimize
PHASES = ['passive_inflation', 'active_contraction', "all"]


# If true, Optimize material parameters, otherswise use the default material parameters
OPTIMIZE_MATPARAMS = True

GAMMA_INC_LIMIT = 0.02

# Max size of gamma
MAX_GAMMA = 0.9


# Optimization method
# DEFAULT_OPTIMIZATION_METHOD = "SLSQP"
OPTIMIZATION_METHODS = ["TNC", "L-BFGS-B", "SLSQP", "ipopt"]


############### SYNTHETIC DATA  #####################
NSYNTH_POINTS = 7
SYNTH_PASSIVE_FILLING = 3

############### LABELS AND NAMES #####################

STRAIN_REGIONS = {"LVBasalAnterior": 1,
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
		  "LVApex": 17}

STRAIN_DIRECTIONS = ["RadialStrain", "LongitudinalStrain", 
                     "CircumferentialStrain", "AreaStrain"]
# Strain regions
STRAIN_REGION_NUMS = STRAIN_REGIONS.values()
STRAIN_REGION_NUMS.sort()
STRAIN_NUM_TO_KEY = {0:"circumferential",
                     1: "radial",
                     2: "longitudinal"}



PASSIVE_INFLATION_GROUP = "passive_inflation"
CONTRACTION_POINT = "contract_point_{}"
ACTIVE_CONTRACTION = "active_contraction"
ACTIVE_CONTRACTION_GROUP = "/".join([ACTIVE_CONTRACTION, CONTRACTION_POINT])
# PASSIVE_INFLATION = "passive_inflation"

############## DIRECTORIES AND PATHS #################3


# Folders and path for which the data is stored in .h5 format
curdir = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SIMULATION_FILE = os.path.join(curdir,'local_results/results.h5')

########## DOLFIN PARAMETERS ############################


# Nonlinear solver
NONLINSOLVER = "snes"

# Nonlinear method 
#(Dolfin Adjoint version < 1.6 newtontr/ls are the only one working)
SNES_SOLVER_METHOD = "newtontr"

# Maximum number of iterations
SNES_SOLVER_MAXITR = 15

# Absolute Tolerance
SNES_SOLVER_ABSTOL = 1.0e-5

# Linear solver "
SNES_SOLVER_LINSOLVER = "lu"
#SNES_SOLVER_LINSOLVER = "mumps"
SNES_SOLVER_PRECONDITIONER = "default"

# Print Non linear solver output
VIEW_NLS_CONVERGENCE = False

