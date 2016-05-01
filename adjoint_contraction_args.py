from patient_data.scripts.data import STRAIN_REGIONS, STRAIN_DIRECTIONS
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
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(level)


    formatter = logging.Formatter('%(message)s') #'\n%(name)s - %(levelname)s - %(message)s\n'
    ch.setFormatter(formatter)
    

    logger.addHandler(ch)
    logger.addFilter(mpi_filt)

    
    dolfin.set_log_active(True)
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


############# DATA ####################
# The main patient that we are using
DEFAULT_PATIENT = "Impact_p16_i43"

# Resulution of the mesh
RESOLUTION = "med_res"


#Centipascals for our centimeter mesh.
KPA_TO_CPA = 0.1

PATIENT_TYPES = ["impact", "healthy"]
DEFAULT_PATIENT_TYPE = "impact"

############### OPTIMIZATION PARAMETERS ######################
# Strain weights for optimization
# What kind of weighting rule 
WEIGHT_RULES = ["equal", "drift", "peak_value", "combination"]
DEFAULT_WEIGHT_RULE = "equal" 
# Any preferred direction for weighting ("c", "l", "r" or None)
WEIGHT_DIRECTIONS = ["c", "l", "r", "all"]
DEFAULT_WEIGHT_DIRECTION = "all"

#Spring constant for base.
BASE_K = 1.0

# Regularization patameter
REG_PAR = 0.001

# Weighting of strain and volume (0=Strain only, 1=Volume only)
ALPHA = 0.5
ALPHA_MATPARAMS = 1.0

# Use the original strain or the deintegrated ones
# USE_DEINTEGRATED_STRAINS = True
# if USE_DEINTEGRATED_STRAINS:
    # from strain_projection.project_strains import STRAIN_FIELDS_PATH


# The different phases we can optimize
PHASES = ['passive_inflation', 'active_contraction', "all"]

# Initial material parameters
INITIAL_MATPARAMS = [0.795, 6.855, 21.207, 40.545] 
# INITIAL_MATPARAMS = [0.291, 2.582, 5,5] 

# If true, Optimize material parameters, otherswise use the default material parameters
OPTIMIZE_MATPARAMS = True

# Scale for optimization algorithm
SCALE = 1.0

# MAX size of gamma steps.
GAMMA_INC_LIMIT = 0.02

# Max size of gamma
MAX_GAMMA = 0.9

# MAX size of pressure step (cPa)
PRESSURE_INC_LIMIT = 0.4

# Optimization method
DEFAULT_OPTIMIZATION_METHOD = "SLSQP"
OPTIMIZATION_METHODS = ["TNC", "L-BFGS-B", "SLSQP"]

# Optimization tolerance 
OPTIMIZATION_TOLERANCE_GAMMA = 1.0e-6
OPTIMIZATION_TOLERANCE_MATPARAMS = 1.0e-9

# Maximum number of iterations
OPTIMIZATION_MAXITER_GAMMA = 100
OPTIMIZATION_MAXITER_MATPARAMS = 30


############### SYNTHETIC DATA  #####################
NSYNTH_POINTS = 7
SYNTH_PASSIVE_FILLING = 3

############### LABELS AND NAMES #####################

# Strain regions
STRAIN_REGION_NUMS = STRAIN_REGIONS.values()
STRAIN_REGION_NUMS.sort()
STRAIN_NUM_TO_KEY = {0:"circumferential",
                     1: "radial",
                     2: "longitudinal"}



PASSIVE_INFLATION_GROUP = "alpha_{}/passive_inflation" 
ACTIVE_CONTRACTION_GROUP = "alpha_{}/active_contraction/contract_point_{}" #.format(alpha, iteration number)
CONTRACTION_POINT = "contract_point_{}"
ALPHA_STR = "alpha_{}"
ACTIVE_CONTRACTION = "active_contraction"
PASSIVE_INFLATION = "passive_inflation"

############## DIRECTORIES AND PATHS #################3


# Folders and path for which the data is stored in .h5 format
curdir = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SIMULATION_FILE = os.path.join(curdir,'local_results/{}/results.h5'.format(DEFAULT_PATIENT))  

########## DOLFIN PARAMETERS ############################


# Nonlinear solver
NONLINSOLVER = "snes"

# Nonlinear method 
#(Dolfin Adjoint version < 1.6 newtontr/ls are the only one working)
SNES_SOLVER_METHOD = "newtontr"

# Maximum number of iterations
SNES_SOLVER_MAXITR = 50

# Absolute Tolerance
SNES_SOLVER_ABSTOL = 1.0e-5

# Linear solver "
SNES_SOLVER_LINSOLVER = "lu"
#SNES_SOLVER_LINSOLVER = "mumps"
SNES_SOLVER_PRECONDITIONER = "default"

OPTIMIZATION_METHOD = "SLSQP"

# Print Non linear solver output
VIEW_NLS_CONVERGENCE = False

