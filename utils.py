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
from adjoint_contraction_args import  logger, PHASES
from pprint import pformat


class UnableToChangePressureExeption(Exception):
    pass

def test():
    raise UnableToChangePressureExeption("test")

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
def print_head(for_res, display_iter = True):
        
    targets = for_res["optimization_targets"]
    reg  = for_res["regularization"]
    keys = targets.keys()+["regularization"]
    n = len(keys)

    head = "\n{:<6}\t".format("Iter") if display_iter else "\n"+" "*7
    head += "{:<7}\t".format("I_tot") + \
           "\t"+(n*"I_{:<10}\t").format(*keys)
    return head

def print_line(for_res, it = None):
    
    func_value = for_res["func_value"]
    targets = for_res["optimization_targets"]
    reg  = for_res["regularization"]
    values = [sum(t.results["func_value"]) for t in targets.values()] + \
             [sum(reg.results["func_value"])]
    n = len(values)
    line = "{:<6d}\t".format(it) if it is not None else ""
    line += "{:<7.2e}".format(func_value) + \
           "\t"+(n*"{:<10.2e}\t").format(*values)
    return line

def passive_inflation_exists(params):
    import h5py, os
    from adjoint_contraction_args import PASSIVE_INFLATION_GROUP

    if not os.path.exists(params["sim_file"]):
        return False
    
    h5file = h5py.File(params["sim_file"])
    key = PASSIVE_INFLATION_GROUP

    # Check if pv point is already computed
    if key in h5file.keys():
        logger.info(Text.green("Passive inflation, {}".format("fetched from database")))
        h5file.close()
        return True
    logger.info(Text.blue("Passive inflation, {}".format("Run Optimization")))
    h5file.close()
    return False

def contract_point_exists(params):
    import h5py, os
    import numpy as np
    from adjoint_contraction_args import ACTIVE_CONTRACTION, CONTRACTION_POINT, PASSIVE_INFLATION_GROUP, PHASES
    
    if not os.path.exists(params["sim_file"]):
        logger.info(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")
        return False

    h5file = h5py.File(params["sim_file"])
    key1 = ACTIVE_CONTRACTION
    key2  = CONTRACTION_POINT.format(params["active_contraction_iteration_number"])
    key3 = PASSIVE_INFLATION_GROUP
    
	
    if not key3 in h5file.keys():
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
        if key1 in h5file.keys() and key2 in h5file[key1].keys():
            pressure = np.array(h5file[key1][key2]["bcs"]["pressure"])[-1]
            logger.info(Text.green("Contract point {}, pressure = {:.3f} {}".format(params["active_contraction_iteration_number"],
                                                                                    pressure, "fetched from database")))
            h5file.close()
            return True
        logger.info(Text.blue("Contract point {}, {}".format(params["active_contraction_iteration_number"],"Run Optimization")))
        h5file.close()
        return False
    except KeyError:
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

def QuadratureSpace(mesh, degree, dim=3):
    import dolfin as d
    if d.DOLFIN_VERSION_MAJOR > 1.6:
        if dim == 1:
            element = d.FiniteElement(family = "Quadrature",
                                        cell = mesh.ufl_cell(),
                                        degree = 4,
                                        quad_scheme="default")
        else:
            element = d.VectorElement(family = "Quadrature",
                                        cell = mesh.ufl_cell(),
                                        degree = 4,
                                        quad_scheme="default")
        
        return d.FunctionSpace(mesh, element)
    else:
        if dim == 1:
            return d.FunctionSpace(mesh, "Quadrature", 4)
        else:
            return d.VectorFunctionSpace(mesh, "Quadrature", 4)

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

    def __init__(self, fldmap, fancyhead = False):

        if fancyhead:
            q = [int(a.split(".")[0]) for a in fldmap[1::2]]
            
            fmt  = '\t'.join(['{:' + '{}'.format(fmt) + '}' \
                              for fmt in q ])

            self.head = fmt.format(*fldmap[0::2])
        else:
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
    pass
