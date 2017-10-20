# Pulse-Adjoint #
 
This repo contains the code that is based on the following papers:

Balaban, G., H. Finsberg, J. Sundnes, M. E. Rognes, H.-H. Odland, S. Ross, and S. T. Wall, 2016: High resolution data assimilation of cardiac mechanics
applied to a dyssynchronous ventricle. International Journal for Numerical Methods in Engineering, 79, no. 11, 1309–1331

Finsberg, H., G. Balaban, S. Ross, T. F. Håland, H. H. Odland, J. Sundnes, and S. Wall, 2017: Estimating cardiac contraction through high resolution
data assimilation of a personalized mechanical model. Journal of Computational Science.


A cardiac computational model is constrained using clinical measurements such as pressure, volume and regional strain. The problem is formulated as a PDE-constrained optimisation problem where the objective functional represents the misfit between measured and simulated data. There are two phases; passive and active. In the passive phase the material parameters are the control parameters, and in the active phase the contraction parameter is the control parameter. The control parameters can be scalar or spatially resolved. The problem is solved using a gradient based optimization algorithm where the gradient is provided by solving the adjoint system.

## Installation ##
```
python setup.py install
```
or change `PREFIX` in the `Makefile` to where you want to install, and run `make install`

## Requirements ##
In order to simply run the code you need
```
* FEniCS version 2016.x
  -- http://fenicsproject.org
* Dolfin-Adjoint version 2016.x
  -- http://www.dolfin-adjoint.org/en/latest/download/index.html
* yaml, h5py
  -- pip install pyyaml, h5py
* mesh_generation
  -- Used for loading the meshes in patient_data
  -- git clone git@bitbucket.org:finsberg/mesh_generation.git

```

## Structure of the repo ##
The main code is found in `pulse_adjoint` folder. There are also currently four submodules: `models` contains the different models for passive and active behaviour, `unloading` contains the code to perform the unloading, `patient_data` is for handling input data and `postprocess` is for postprocessing the results. 

The starting point should be the demos in the `demo` folder. It might also be useful to look at the different tests in the `test` folder.


## Doumentation ##
Documentation is found here: `http://pulse-adjoint.readthedocs.io/en/latest/`
To create your own documentation, do
```
cd doc
make html
```


## License ##
c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED

Authors: Henrik Finsberg

END-USER LICENSE AGREEMENT

PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this software you agree with the terms and 
conditions of this license agreement. If you do not accept the terms of this license agreement you may 
not install or use this software.

Permission to use, copy, modify and distribute any part of this software for non-profit educational 
and research purposes, without fee, and without a written agreement is hereby granted, provided that 
the above copyright notice, and this license agreement in its entirety appear in all copies. Those desiring 
to use this software for commercial purposes should contact Simula Research Laboratory AS: post@simula.no 

IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, 
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE 
"PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE SOFTWARE PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION 
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.  SIMULA RESEARCH LABORATORY MAKES NO 
REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT NOT LIMITED 
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS

## Contributors ##
* Henrik Finsberg (henriknf@simula.no)