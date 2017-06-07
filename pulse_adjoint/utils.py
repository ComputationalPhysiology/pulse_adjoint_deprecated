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
from pprint import pformat
from .adjoint_contraction_args import  logger, PHASES




def get_dimesion(u):
    from dolfin import DOLFIN_VERSION_MAJOR
    from ufl.domain import find_geometric_dimension
    
    if DOLFIN_VERSION_MAJOR > 1.6:
        dim = find_geometric_dimension(u)
    else:
        dim = u.geometric_dimension()

    return dim


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
    head += "{:<10}\t".format("Obj") + \
            "{:<10}".format("||grad||") + \
           "\t"+(n*"I_{:<10}\t").format(*keys) 
    
    return head

def print_line(for_res, it = None, grad_norm = None, func_value = None):
    
    func_value = for_res["func_value"] if func_value is None else func_value
    grad_norm = 0.0 if grad_norm is  None else grad_norm

    targets = for_res["target_values"]
    # reg  = for_res["regularization"]
    
    reg_func = targets.pop("regularization")
    values = targets.values() + [reg_func]
    targets["regularization"] = reg_func
    
    n = len(values)
    line = "{:<6d}\t".format(it) if it is not None else ""
    line += "{:<10.2e}\t".format(func_value) + \
            "{:<10.2e}".format(grad_norm) + \
           "\t"+(n*"{:<10.2e}\t").format(*values)
    return line

def passive_inflation_exists(params):
    import h5py, os
    from .adjoint_contraction_args import PASSIVE_INFLATION_GROUP

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

def check_group_exists(h5name, h5group):
    import h5py, os

    if not os.path.exists(h5name):
        return False

    try:
        h5file = h5py.File(h5name)
    except:
        return False

    group_exists = False
    if h5group in h5file:
        group_exists = True
        
    h5file.close()
    return group_exists
        

def contract_point_exists(params):
    import h5py, os
    import numpy as np
    from .adjoint_contraction_args import ACTIVE_CONTRACTION, CONTRACTION_POINT, PASSIVE_INFLATION_GROUP, PHASES
    
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

def get_simulated_pressure(params):
    """
    Get the last simulated pressure stored in
    the result file specified by given parameters

    :param dict params: adjoint contracion parameters
    :returns: The final pressure
    :rtype: float

    """
    

    import h5py, numpy
    from .adjoint_contraction_args import ACTIVE_CONTRACTION, CONTRACTION_POINT, PASSIVE_INFLATION_GROUP, PHASES
    
    key1 = ACTIVE_CONTRACTION
    key2  = CONTRACTION_POINT.format(params["active_contraction_iteration_number"])
    key3 = PASSIVE_INFLATION_GROUP
            
    with h5py.File(params["sim_file"], "r") as h5file:
        try:
            pressure = numpy.array(h5file[key1][key2]["bcs"]["pressure"])[-1]
        except:
            pressure = None
            
    return pressure

def list_sum(l):
    """
    Return the sum of a list, when the convetiional
    method (like `sum`) it not working.
    For example if you have a list of dolfin functions.

    :param list l: a list of objects 
    :returns: The sum of the list. The type depends on 
              the type of elemets in the list

    """
    
    if not isinstance(l, list):
        return l

    out = l[0]
    for item in l[1:]:
        out += item
    return out

        
def rename_attribute(object_, old_attribute_name, new_attribute_name):
    setattr(object_, new_attribute_name,
            getattr(object_, old_attribute_name))
    delattr(object_, old_attribute_name)


def get_spaces(mesh):
    """
    Return an object of dolfin FunctionSpace, to 
    be used in the optimization pipeline

    :param mesh: The mesh
    :type mesh: :py:class:`dolfin.Mesh`
    :returns: An object of functionspaces
    :rtype: object

    """
    
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
    """
    From FEniCS version 1.6 to 2016.1 there was a change in how 
    FunctionSpace is defined for quadrature spaces.
    This functions checks your dolfin version and returns the correct
    quadrature space

    :param mesh: The mesh
    :type mesh: :py:class:`dolfin.Mesh`
    :param int degree: The degree of the element 
    :param int dim: For a mesh of topological dimension 3, 
                    dim = 1 would be a scalar function, and 
                    dim = 3 would be a vector function. 
    :returns: The quadrature space
    :rtype: :py:class:`dolfin.FunctionSpace`

    """
    
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
    
    **Example of use**::

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
