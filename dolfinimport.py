"""This module handles all dolfin import in pulse_adjoint. Here dolfin and
dolfin_adjoint gets imported. If dolfin_adjoint is not present it will not
be imported."""


from dolfin import *
import dolfin


try:
    import dolfin_adjoint
    from dolfin_adjoint import *
    

except:
    # FIXME: Should we raise some sort of warning?
    dolfin_adjoint = None
    pass
