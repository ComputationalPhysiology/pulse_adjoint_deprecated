import numpy as np
from math import *

def ca_transient_orig(t, tau1 = 20.0, tau2 = 30.0, amp=1):
    """
    This is a more realistic form of a Calcium transient
    """

    tstart = 0.0
    ca_diast = 0.0
    ca_ampl = amp

    beta = (tau1/tau2)**(-1/(tau1/tau2-1))-(tau1/tau2)**(-1/(1-tau2/tau1));

    if np.isscalar(t):

        if (t < tstart):
	    ca = ca_diast
        else:
	    ca = (ca_ampl-ca_diast)/beta*(exp(-(t-tstart)/tau1)-exp(-(t-tstart)/tau2))+ca_diast

        
    else:

        assert hasattr(t, "__len__"), "t must be scalar of list, got {}".format(type(t))
        ca  = np.zeros(len(t))
        for i, ti in enumerate(t):

            if (ti < tstart):
	        ca[i] = ca_diast
            else:
	        ca[i] = (ca_ampl-ca_diast)/beta*(exp(-(ti-tstart)/tau1)-exp(-(ti-tstart)/tau2))+ca_diast
                
    return ca


def ca_transient(t, amp = 1.0):
    """
    This is just a sum of two gaussian fuctions
    """
    
    b=75
    c=45
    res = amp*(exp( -((t-b)*(t-b)) / (2.0*c*c) ) - exp( -((0-b)*(0-b)) / (2.0*c*c) ))
    return res

if __name__ == "__main__":
    t = np.linspace(0,200,200)
    ca = np.zeros(t.shape)
    l =  np.zeros(t.shape)

    for i in range(len(t)):
        ca[i] = ca_transient_orig(t[i], amp = 45e3)
        l[i] = ca_transient(t[i], amp = 60e3)

    print ca_transient(3)
    print liu(3)

    import matplotlib.pyplot as plt

    plt.plot(t,ca, label = "calcium")
    plt.plot(t,l, label = "liu")
    plt.legend()
    plt.show()
	
