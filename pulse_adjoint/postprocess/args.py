import dolfin
import numpy as np
from pulse_adjoint.setup_optimization import RegionalParameter
from pulse_adjoint.adjoint_contraction_args import *

ALL_ACTIVE_GROUP = "alpha_{}/reg_par_{}/active_contraction/contract_point_{}"
ALL_PASSIVE_GROUP = "alpha_{}/reg_par_0.0/passive_inflation"

STRAIN_REGION_NAMES = {1:"Anterior",
                       2:"Anteroseptal",
                       3:"Septum",
                       4:"Inferior",
                       5:"Posterior",
                       6:"Lateral",
                       7:"Anterior",
                       8:"Anteroseptal",
                       9:"Septum",
                       10:"Inferior",
                       11:"Posterior",
                       12:"Lateral",
                       13:"Anterior",
                       14:"Septum",
                       15:"Inferior",
                       16:"Lateral",
                       17:"Apex"}

STRAIN_REGIONS = {1:"LVBasalAnterior",
                 2:"LVBasalAnteroseptal",
                 3:"LVBasalSeptum",
                 4:"LVBasalInferior",
                 5:"LVBasalPosterior",
                 6:"LVBasalLateral",
                 7:"LVMidAnterior",
                 8:"LVMidAnteroseptal",
                 9:"LVMidSeptum",
                 10:"LVMidInferior",
                 11:"LVMidPosterior",
                 12:"LVMidLateral",
                 13:"LVApicalAnterior",
                 14:"LVApicalSeptum",
                 15:"LVApicalInferior",
                 16:"LVApicalLateral",
                 17:"LVApex"}

dolfin.parameters["allow_extrapolation"] = True
work_pairs = ["SE", "PF", "pgradu", "strain_energy"]
cases = ["full", "comp_fiber", "comp_long"]
