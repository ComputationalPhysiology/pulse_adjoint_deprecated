from . import adjoint_contraction_args as args

from . import forward_runner
from . import setup_optimization
from . import run_optimization
from . import utils
from . import numpy_mpi
from . import heart_problem
from . import lvsolver
from . import optimal_control


# Subpackages
from . import models
from . import postprocess
from . import unloading
from . import io
from . import patient_data

from .patient_data import Patient, FullPatient, LVTestPatient, BiVTestPatient

from .iterate import iterate
from .adjoint_contraction_args import logger
from .setup_optimization import RegionalParameter


from .kinematics import (
    SecondOrderIdentity,
    DeformationGradient,
    Jacobian,
    GreenLagrangeStrain,
    LeftCauchyGreen,
    RightCauchyGreen,
    EulerAlmansiStrain,
    Invariants,
    PiolaTransform,
    InversePiolaTransform,
)


__version__ = "1.0"
__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
