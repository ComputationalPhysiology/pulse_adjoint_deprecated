from pulse_adjoint.unloading import *
from pulse_adjoint.setup_parameters import setup_general_parameters
setup_general_parameters()

from pulse_adjoint import LVTestPatient
geo_lv = LVTestPatient()
p_lv= 2.0

from pulse_adjoint import BiVTestPatient
geo_biv = BiVTestPatient()
p_biv= (2.0, 1.0)


def test_fixed_point_lv():

    unloader = FixedPoint(geo_lv, p_lv,
                          h5name = "fixedpoint_lv.h5",
                          options = {"maxiter":15})
    unloader.unload()

def test_raghavan_lv():
    unloader = Raghavan(geo_lv, p_lv,
                        h5name = "raghavan_biv.h5",
                        options = {"maxiter":1})
    unloader.unload()


def test_hybrid_lv():
 
    unloader = Hybrid(geo_lv, p_lv,
                      h5name = "hybrid_biv.h5",
                      options = {"maxiter":1})
    unloader.unload()


def test_fixed_point_biv():

    unloader = FixedPoint(geo_biv, p_biv,
                          h5name = "fixedpoint_biv.h5",
                          options = {"maxiter":1})
    unloader.unload()
 

def test_raghavan_biv():

    unloader = Raghavan(geo_biv, p_biv,
                        h5name = "raghavan_biv.h5",
                        options = {"maxiter":1})
    unloader.unload()
    
    
def test_hybrid_biv():

  
    unloader = Hybrid(geo_biv, p_biv,
                      h5name = "hybrid_biv.h5",
                      options = {"maxiter":1})
    unloader.unload()
    
    

    

if __name__ == "__main__":
    
    test_fixed_point_lv()
    # test_fixed_point_biv()
    # test_raghavan_lv()
    # test_raghavan_biv()
    # test_hybrid_biv()
