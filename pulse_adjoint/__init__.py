import adjoint_contraction_args as args

import forward_runner
import setup_optimization
import run_optimization
import utils
import pa_io
import numpy_mpi
import heart_problem
import lvsolver
import optimal_control

import models
import postprocess

from iterate import iterate
from adjoint_contraction_args import logger
from setup_optimization import RegionalParameter


from kinematics import (SecondOrderIdentity,
                        DeformationGradient,
                        Jacobian,
                        GreenLagrangeStrain,
                        LeftCauchyGreen,
                        RightCauchyGreen,
                        EulerAlmansiStrain,
                        Invariants,
                        PiolaTransform,
                        InversePiolaTransform)


__version__ = '1.0'
__author__  = 'Henrik Finsberg'
__credits__ = ['Henrik Finsberg']
__license__ = 'LGPL-3'
__maintainer__ = 'Henrik Finsberg'
__email__ = 'henriknf@simula.no'
