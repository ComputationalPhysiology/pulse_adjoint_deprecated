# Pulse-Adjoint #

This repo contains the code that is based on the following paper: (link to paper)

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
```

### Optional ###
To get the most out of it you also need
```
* yaml
  -- Used for storing parameters
  -- pip install pyyaml
* mesh_generation
  -- Used for loading the meshes in patient_data
  -- git clone git@bitbucket.org:finsberg/mesh_generation.git

```

## Structure of the repo ##
The main code is found in `pulse_adjoint` folder. There are also currently four submodules: `models` contains the different models for passive and active behaviour, `unloading` contains the code to perform the unloading, `patient_data` is for handling input data and `postprocess` is for postprocessing the results. 

The starting point should be the demos in the `demo` folder. It might also be useful to look at the different tests in the `test` folder.


## License ##
PULSE-ADJOINT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PULSE-ADJOINT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with PULSE-ADJOINT. If not, see <http://www.gnu.org/licenses/>.

## Contributors ##
* Henrik Finsberg (henriknf@simula.no)