[![Documentation
Status](https://readthedocs.org/projects/pulse-adjoint/badge/?version=latest)](https://pulse-adjoint.readthedocs.io/en/latest/?badge=latest)

# Pulse-Adjoint

Here we demostrate how you can constrain a cardiac computational model
using clinical measurements such as pressure, volume and regional
strain. The problem is formulated as a PDE-constrained optimisation
problem where the objective functional represents the misfit between
measured and simulated data. There are two phases; passive and active. In the
passive phase the material parameters are the control parameters, and
in the active phase the contraction parameter is the control
parameter. The control parameters can be scalar or spatially
resolved. The problem is solved using a gradient based optimization
algorithm where the gradient is provided by solving the adjoint
system. 

## Note
The code here is currently undergoing a major refactoring in order for
it to be compatible with more recent versions of fenics as well as the
standalone cardiac mechanics solver
[pulse](https://github.com/ComputationalPhysiology/pulse). 
We expect the `master` branch to be stable, as development will mainly
be done on other branches.
The original code, hosted at
[Bitbucket](https://bitbucket.org/finsberg/pulse_adjoint) is no longer
maintained.

## Installation
`pulse-adjoint` is written in pure python but based on `fenics` and
`dolfin-adjoint` (using `libabjoint`). Therefore `fenics` version 2017
and `dolfin-adjoint` that uses `libadjoint` (not `pyadjoint`) is
currently the only supported versions. You also need to install
[`pulse`](https://github.com/ComputationalPhysiology/pulse). 

As `pulse-adjoint` is a pure python package you can install it by
simply typing 
```
python setup.py install
```
when you are in the same folder as the `setup.py`.

## Doumentation ##
Documentation is found at [pulse-adjoint.readthedocs.io](http://pulse-adjoint.readthedocs.io/en/latest)


## Citing ##

You are welcomed to use this code for your own reaseach, but encourage
you to cite the following paper:


>Finsberg, H., Xi, C., Tan, J.L., Zhong, L., Genet, M., Sundnes, J.,
>Lee, L.C. and Wall, S.T., 2018. Efficient estimation of personalized
>biventricular mechanical function employing gradient‐based
>optimization. International journal for numerical methods in
>biomedical engineering,
>p.e2982. [DOI](https://doi.org/10.1002/cnm.2982)


## License ##
GNU LGPL v3

## Contributors ##
* Henrik Finsberg (henriknf@simula.no)
