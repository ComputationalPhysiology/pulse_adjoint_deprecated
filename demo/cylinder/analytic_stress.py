"""
This are the analytic stresses for a cylinder 
with a linear elastic material
"""

import numpy as np
import matplotlib.pyplot as plt

from cylinder import load_mesh, r_inner, t

def circ_stress(P, r):

    r_outer = r_inner + t

    return P * r_inner**2 / (r_outer**2 - r_inner**2) + P * r_inner**2 * r_outer**2 / (r**2*(r_outer**2 - r_inner**2) )

def rad_stress(P, r):

    r_outer = r_inner + t

    return P * r_inner**2 / (r_outer**2 - r_inner**2) - P * r_inner**2 * r_outer**2 / (r**2*(r_outer**2 - r_inner**2) )


        

if __name__ == "__main__":
    P = 10.0
    Tc = []
    Tr = []
    R = np.linspace(r_inner, 2)
    for r in R:
        Tc.append(circ_stress(P, r))
        Tr.append(rad_stress(P, r))

    plt.plot(R, Tc, label = "circ")
    plt.plot(R, Tr, label = "rad")
    plt.show()

    
