# CardioMechanical Patient-Specific Simulator (campass)#

This repo contains the code that is based on the following paper: (link to paper)
Name might be changed later (suggestions?)

A cardiac computational model is constrained using clinical measurements such as pressure, volume and regional strain. The problem is formulated as a PDE-constrained optimisation problem where the objective functional represents the misfit between measured and simulated data. The control parameter for the active phase is a spatially varying contraction parameter defined at every vertex in the mesh. The control parameters for the passive phase is material parameters for a Holzapfel and Ogden transversally isotropic material. The problem is solved using a gradient based optimization algorithm where the gradient is provided by solving the adjoint system.

## Requirements ##
In order to simply run the code you need
```
* FEniCS version >= 1.6
  -- http://fenicsproject.org
* Dolfin-Adjoint version >= 1.6
  -- http://www.dolfin-adjoint.org/en/latest/download/index.html
* hao_solver (Should perhaps make this (at least a basic version) as a part of the repo)
  -- Solver for the Holzapfel and Odgen material
  -- git clone git@bitbucket.org:Gabrielbalaban/haosolver.git
  -- Ask Gabriel for access
* numpy_mpi (Should perhaps make this as a part of the repo)
  -- Scripts to make in run in parallell
  -- git clone git@bitbucket.org:Gabrielbalaban/numpy_mpi.git
  -- Ask Gabriel for access
```
Optional
To get the most out of it you also need
```
* yaml
  -- Used for storing parameters
  -- pip install pyyaml
* patient_data (should make a basic version part of the repo)
  -- Used for loading patient data in the correct format
  -- This repo also contains some test data
  -- git clone git@bitbucket.org:finsberg/patient_data.git
  -- Ask Henrik for access
* mesh_generation
  -- Used for loading the meshes in patient_data
  -- git clone git@bitbucket.org:finsberg/mesh_generation.git
  -- Ask Henrik for access

```

## Structure of the repo ##

To setup the optimization you want to change the parameters 
in setup_optimization.py. Here you should specify the paths to the data
you are using and what type of parameters/setups you want to use. 

The main script to run is run_full_optimization.py which optimizes both
material parameters for the passive phase and contraction parameters for the
active phase for all points in your data sets. 
This script calls the functions from run_optimization.py where the different phases
are separated.

There are two important classes to be aware of. One is the HeartProblem class 
located in heart_problem.py which is used as a communicator with the solver.
The other is the ForwardRunner class which is used to run the forward problem
and is located in forward_runner.py

### Other features ###

* There is written some basic tests in tests/
which is used to tests that the adjoint calculations are correct.
* One can generate synthetic data and test the optimization using this data
as input. To do so run the script synthetic_data.py

## Contributors ##

* Henrik Finsberg and Gabriel Balaban