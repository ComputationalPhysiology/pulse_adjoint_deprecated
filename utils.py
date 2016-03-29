#!/usr/bin/env python

from dolfin import mpi_comm_world
from pprint import pformat
from adjoint_contraction_args import ALPHA_STR, ACTIVE_CONTRACTION, CONTRACTION_POINT, PASSIVE_INFLATION, logger, PHASES
from numpy_mpi import *
import sys, os,  matplotlib, h5py



# Dummy object
class Object(object):pass

def print_optimization_report(params, opt_controls, ini_for_res, for_result_opt, opt_result = None):

    if opt_result:
        logger.info("Optimization terminated...")
        logger.info("\tExit status {}".format(opt_result["status"]))
        logger.info("\tSuccess: {}".format(opt_result["success"]))
        logger.info("\tMessage: {}".format(opt_result["message"]))
        logger.info("\tFunction Evaluations: {}".format(opt_result["nfev"]))
        logger.info("\tGradient Evaluations: {}".format(opt_result["njev"]))
        logger.info("\tNumber of iterations: {}".format(opt_result["nit"]))

    logger.info("\nFunctional Values")
    logger.info(" "*8 + "Strain" + " "*5 + "Volume")
    logger.info("initial " + "{:.5f}".format(ini_for_res.func_value_strain) \
                + " "*5 + "{:.5f}".format(ini_for_res.func_value_volume))
    logger.info("optimal " + "{:.5f}".format(for_result_opt.func_value_strain) \
                + " "*5 + "{:.5f}".format(for_result_opt.func_value_volume))

    if params["phase"] == PHASES[0]:
        logger.info("\nMaterial Parameters")
        logger.info("Initial {}".format(params["Material_parameters"].values()))
        logger.info("Optimal {}".format(gather_broadcast(opt_controls.vector().array())))
    else:
        pass

def passive_inflation_exists(params):

    if not os.path.exists(params["sim_file"]):
        return False
    
    h5file = h5py.File(params["sim_file"])
    key1 = ALPHA_STR.format(params["alpha_matparams"])
    key2 = PASSIVE_INFLATION

    # Check if pv point is already computed
    if key1 in h5file.keys() and key2 in h5file[key1].keys():
        logger.info(Text.green("Passive inflation, alpha = {} {}".format(params["alpha_matparams"],"fetched from database")))
        h5file.close()
        return True
    logger.info(Text.blue("Passive inflation, alpha = {} {}".format(params["alpha_matparams"],"Run Optimization")))
    h5file.close()
    return False

def contract_point_exists(params):
    
    if not os.path.exists(params["sim_file"]):
        mpi_print(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")
        return False

    h5file = h5py.File(params["sim_file"])
    key1 = ALPHA_STR.format(params["alpha"])
    key2 = ACTIVE_CONTRACTION
    key3  = CONTRACTION_POINT.format(params["active_contraction_iteration_number"])
    key4 = PASSIVE_INFLATION
    key5 = ALPHA_STR.format(params["alpha_matparams"])
	
    if not key5 in h5file.keys() or key4 not in h5file[key5].keys():
        mpi_print(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")

    
    if params["phase"] == PHASES[0]:
        h5file.close()
        return False
    
    if not key1 in h5file.keys():
        h5file.close()
        return False

   
    try:

        # Check if pv point is already computed
        if key2 in h5file[key1].keys() and key3 in h5file[key1][key2].keys():
            pressure = np.array(h5file[key1][key2][key3]["lv_pressures"])[0]
            mpi_print(Text.green("Contract point {}, alpha = {} pressure = {:.3f} {}".format(params["active_contraction_iteration_number"],
                                                                           params["alpha"], pressure, "fetched from database")))
            h5file.close()
            return True
        mpi_print(Text.blue("Contract point {}, alpha = {} {}".format(params["active_contraction_iteration_number"], params["alpha"], "Run Optimization")))
        h5file.close()
        return False
    except:
        return False

def list_sum(l):
    if not isinstance(l, list):
        return l

    out = l[0]
    for item in l[1:]:
        out += item
    return out

        

def setup_matplotlib():
    matplotlib.rcParams.update({'figure.autolayout': True})
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 26} 

    matplotlib.rc('font', **font)
    matplotlib.pyplot.rc('text', usetex=True)
    matplotlib.rcParams['text.usetex']=True
    matplotlib.rcParams['text.latex.unicode']=True



def get_spaces(mesh):
    from dolfin import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace
    
    # Make a dummy object
    spaces = Object()

    # A real space with scalars used for dolfin adjoint   
    spaces.r_space = FunctionSpace(mesh, "R", 0)
    
    # A space for the strain fields
    spaces.strainfieldspace = VectorFunctionSpace(mesh, "CG", 1, dim = 3)

    # A space used for scalar strains
    spaces.strainspace = VectorFunctionSpace(mesh, "R", 0, dim = 3)

    # Spaces for the strain weights
    spaces.strain_weight_space = TensorFunctionSpace(mesh, "R", 0)
    
    return spaces

class Text:
    """
    Ansi escape sequences for coloured text output
    """
    _PURPLE = '\033[95m'
    _OKBLUE = '\033[94m'
    _OKGREEN = '\033[92m'
    _YELLOW = '\033[93m'
    _RED = '\033[91m '
    _ENDC = '\033[0m'
    
    @staticmethod
    def blue(text):
        out = Text._OKBLUE + text  + Text._ENDC
        return out
    
    @staticmethod
    def green(text):
        out = Text._OKGREEN +  text  + Text._ENDC
        return out
    
    @staticmethod
    def red(text):
        out = Text._RED + text + Text._ENDC
        return out
    
    @staticmethod
    def yellow(text):
        out = Text._YELLOW + text + Text._ENDC
        return out
    
    @staticmethod
    def purple(text):
        out = Text._PURPLE + text + Text._ENDC
        return out
    
    @staticmethod
    def decolour(text):
        to_remove = [Text._ENDC,
                     Text._OKBLUE,
                     Text._OKGREEN,
                     Text._RED,
                     Text._YELLOW,
                     Text._PURPLE]
        
        for chars in to_remove:
            text = text.replace(chars,"")
        return text

if __name__ == '__main__':
    # logger = make_logger("test_logger", INFO)
    # logger.info("testmessage")
    # logger = make_logger("test_logger", INFO)
    # logger.info("testmessage")
    gamma = range(18)
    print_regional_gamma(gamma)
