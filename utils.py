#!/usr/bin/env python
from adjoint_contraction_args import  logger, PHASES
from pprint import pformat


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

# Dummy object
class Object(object):pass

def print_optimization_report(params, opt_controls, init_controls, 
                              ini_for_res, opt_for_res, opt_result = None):

    from numpy_mpi import gather_broadcast

    if opt_result:
        logger.info("\nOptimization terminated...")
        logger.info("\tExit status {}".format(opt_result["status"]))
        # logger.info("\tSuccess: {}".format(opt_result["success"]))
        logger.info("\tMessage: {}".format(opt_result["message"]))
        logger.info("\tFunction Evaluations: {}".format(opt_result["nfev"]))
        logger.info("\tGradient Evaluations: {}".format(opt_result["njev"]))
        logger.info("\tNumber of iterations: {}".format(opt_result["nit"]))
        logger.info("\tNumber of crashes: {}".format(opt_result["ncrash"]))
        logger.info("\tRun time: {:.2f} seconds".format(opt_result["run_time"]))

    logger.info("\nFunctional Values")
    logger.info("\tTotal\t\tStrain\t\tVolume")
    logger.info("Initial\t{:.2e}\t{:.2e}\t{:.2e}".format(ini_for_res.func_value, 
                                                         ini_for_res.func_value_strain,
                                                         ini_for_res.func_value_volume))
    logger.info("Optimal\t{:.2e}\t{:.2e}\t{:.2e}".format(opt_for_res.func_value, 
                                                         opt_for_res.func_value_strain,
                                                         opt_for_res.func_value_volume))

    if params["phase"] == PHASES[0]:
        logger.info("\nMaterial Parameters")
        logger.info("Initial {}".format(init_controls))
        logger.info("Optimal {}".format(gather_broadcast(opt_controls.vector().array())))
    else:
        logger.info("\nContraction Parameter")
        logger.info("\tMin\tMean\tMax")
        logger.info("Initial\t{:.5f}\t{:.5f}\t{:.5f}".format(init_controls.min(), 
                                                             init_controls.mean(), 
                                                             init_controls.max()))
        opt_controls_arr = gather_broadcast(opt_controls.vector().array())
        logger.info("Optimal\t{:.5f}\t{:.5f}\t{:.5f}".format(opt_controls_arr.min(), 
                                                             opt_controls_arr.mean(), 
                                                             opt_controls_arr.max()))

def passive_inflation_exists(params):
    import h5py, os
    from adjoint_contraction_args import ALPHA_STR, PASSIVE_INFLATION

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
    import h5py, os
    import numpy as np
    from adjoint_contraction_args import ALPHA_STR, ACTIVE_CONTRACTION, CONTRACTION_POINT, PASSIVE_INFLATION, PHASES
    
    if not os.path.exists(params["sim_file"]):
        logger.info(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")
        return False

    h5file = h5py.File(params["sim_file"])
    key1 = ALPHA_STR.format(params["alpha"])
    key2 = ACTIVE_CONTRACTION
    key3  = CONTRACTION_POINT.format(params["active_contraction_iteration_number"])
    key4 = PASSIVE_INFLATION
    key5 = ALPHA_STR.format(params["alpha_matparams"])
	
    if not key5 in h5file.keys() or key4 not in h5file[key5].keys():
        logger.info(Text.red("Run passive inflation before systole"))
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
            logger.info(Text.green("Contract point {}, alpha = {} pressure = {:.3f} {}".format(params["active_contraction_iteration_number"],
                                                                           params["alpha"], pressure, "fetched from database")))
            h5file.close()
            return True
        logger.info(Text.blue("Contract point {}, alpha = {} {}".format(params["active_contraction_iteration_number"], params["alpha"], "Run Optimization")))
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


class TablePrint(object):
    """
    Print output in nice table format.
    Example of use:

      fldmap = (
         'LVP',  '0.5f',
         'LV_Volume', '0.5f',
         'Target_Volume', '0.5f',
         'I_strain', '0.2e',
         'I_volume', '0.2e',
         'I_reg', '0.2e',
         )

      my_print = TablePrint(fldmap)
      print my_print.print_head()
      print my_print.print_line(LVP=1, LV_Volume=1, Target_Volume=1, 
                                I_strain=1, I_volume=1, I_reg=1)

    """

    def __init__(self, fldmap):

        self.head = '\n'+'\t'.join(fldmap[0:len(fldmap):2]) 
        self.fmt  = '\t'.join(['{' + '{0}:{1}'.format(col,fmt) + '}' \
                          for col, fmt in zip(
                              fldmap[0:len(fldmap):2], \
                              fldmap[1:len(fldmap):2] \
                              )])
    def print_head(self):
        return self.head

    def print_line(self, **kwargs):
        return  self.fmt.format(**kwargs)

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
