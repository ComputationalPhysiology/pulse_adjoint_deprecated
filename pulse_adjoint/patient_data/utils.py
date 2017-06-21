#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
import numpy as np
import logging


class Object : pass

log_level = logging.WARNING
def make_logger(name, level = logging.INFO):
    import logging
    import dolfin
    
    mpi_filt = lambda: None
    def log_if_proc0(record):
        if dolfin.mpi_comm_world().rank == 0:
            return 1
        else:
            return 0
    mpi_filt.filter = log_if_proc0

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(0)


    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    

    logger.addHandler(ch)
    logger.addFilter(mpi_filt)

    
    dolfin.set_log_active(False)
    dolfin.set_log_level(dolfin.WARNING)
    
    return logger

logger = make_logger("Patient", log_level)

def m3_2_ml(vol):
    """Convert volume from m3 to ml
    """
    # 1 m3  = 1000 dm3
    # 1 dm3 = 1 L = 1000 ml
    # -> 1m3 = 1000*1000 ml 
    if len(vol) == 1:
	return 1000*1000*vol
    else:
	return [1000*1000*v for v in vol]

def calibrate_strain(trace, ref_time=0, relative_strain = True):
    """
    Takes a strain orientation and strain region and returns the 
    strain tace calculated with index ref_time as reference.
    input_data_older is in a h5py.File format

    The loaded strain traces from Echo uses the first timeslot as reference.
    Let us write 0 = e00, e10, e20, ..., en0 for the strain traces relative to 
    the first time slot.

    The strain taces e0a, e1a, e2a ... ena relative to time_slot a may be found
    though the formula eia = (ei0 - ea0)/(ea0 + 1)

    *Arguments*
      strain_orientation (string)
        LongitudinalStrain, CircumferentialStrain, RadialStrain or AreaStrain
      strain_region (string)
        Name of strain region, eg LVBasalAnterior
      ref_time (int)
        Index between 0 and length(self.times) where you want to start
      realtive_strain (boolean)
        If true, we shift the whole strain trace to start at ref time and compute all 
        strains relative to the strain at ref time.
        This may cause discontinuities in the strain data due to drift.
        If false, we shift the strain trace without recomputing the strain relative to
        the strain at ref time

    *Returns*
      Numpy array of strain values starting at ref_time with zero


    """
    if relative_strain:
        
        new_trace = np.zeros(len(trace))     
        ea0 = trace[ref_time]
        
        for i in range(len(trace)):
            
            ei0 = trace[i]
            eia = (ei0 - ea0)/(ea0 + 1)
            new_trace[i] = eia

            
        st = np.roll(new_trace, -ref_time)      
           
    else:
        st = np.roll(trace, -ref_time)

  
    return st

def correct_drift(y, use_spline = True):

    if isinstance(y, np.ndarray):
        Y = y.tolist()
    else:
        Y = np.copy(y).tolist()

    if use_spline:
        y_extra = extrapolate_to_final_strain(y)
        Y.append(y_extra)

    X = range(len(Y))
    # Create a linear interpolant between the first and the new point
    line = [ i*(Y[-1] - Y[0])/(len(X)-1) for i in X]

    # Subtract the line from the original points
    Y_sub = np.subtract(Y, line)

    if use_spline:
        # The strain trace don't include the new point
        return Y_sub[:-1]
    else:
        return Y_sub


def h5py2dict(hdf):
    """
    Convert h5py file to dictionary recursively
    """
    import h5py
   
    if isinstance(hdf, h5py._hl.group.Group):
        t = {}
        
        for key in hdf.keys():
            t[key] = h5py2dict(hdf[key])
    
                
    elif isinstance(hdf, h5py._hl.dataset.Dataset):
        t = np.array(hdf)

    return t


def compute_strain_weights(strain,
                           rule = "equal",
                           direction = "all",
                           custom_weights = None):

    """Compute weights on the strain regions according to some rule
        
    *Arguments*
    rule (string)
    Either peak value,  drift, combination, equal or custom. If peak value, regions with high strain will 
    have smaller weight. If drift, regions with high drift will have
    higher weight. If combination drift and peak value will both be considered.
    If equal, then all regions will be given the same weight
    If custom then eqaul weigths will be assigned to regions/directions in val
    
    direction (string)
    Either c (circumferential), l (longitidinal), r (radial) or None
    If l, then only the strains in the longitudinal direction will be
    given a weight. All other strains will be given a very small weight
    
    custom_weights (list of tupples)
    Tu be used if rule = custom.
    Example val = [("l",3), ("c", 5)]. Then region 3 and 5 in direction "l"
    and "c" respectively will be given all the weights
    """


    rules = ["equal"]
    assert rule in rules, \
        "Weight rule must be one of {}, given is {}".format(rules, rule)
    
    dirs = ["all", "r", "c", "l"]
    assert direction in dirs, \
        "Weight direction must be one of {}, given is {}".format(dirs, direction)
    
    def normalize(lst, missing_idx, eps):
        weights = np.multiply(lst, missing_idx)
        weights[np.where(missing_idx == 0)] = eps
        return weights
    
    # A small number
    eps = 1e-10
    
        
    nregions = len(strain.keys())
    
    # Use custom weight or set them all to 1
    if np.shape(custom_weights) == (nregions,3):
        weigths = np.array(custom_weights)
    else:
        weigths = np.ones((nregions, 3))
        
            
    missing_idx = find_missing_measurements(strain)

    if direction == "c":
        weigths.T[1,:] = eps
        weigths.T[2,:] = eps
        

    elif direction == "r":
        weigths.T[0,:] = eps
        weigths.T[2,:] = eps
        
    elif direction == "l":
        weigths.T[0,:] = eps
        weigths.T[1,:] = eps
        
    return normalize(weigths, missing_idx, eps)


def get_regional_midpoints(strain_markers, mesh):
    
    coords = {region : {coord:[] for coord in range(3)} for region in range(18)}

    import dolfin
    for cell in dolfin.cells(mesh):
            
        # Get coordinates to cell midpoint
        x = cell.midpoint().x()
        y = cell.midpoint().y()
        z = cell.midpoint().z()

        # Get index of cell
        index = cell.index()

        region = strain_markers.array()[index]
        
        coords[region][0].append(x)
        coords[region][1].append(y)
        coords[region][2].append(z)

    mean_coords = {region : np.zeros(3) for region in range(18)}
    for i in range(18):
        for j in range(3):
            mean_coords[i][j] = np.mean(coords[i][j])
        
    return mean_coords, coords



def find_missing_measurements(strains):
    """Finds the idices of missing strain measurements

    *Arguments*
          strains (dict)
            Dicitionary with original strain measurements

    *Return*
         missing_idx (nregions*3 array)
            Binary array where 0 means missing and 1 is not missing.

    """

    missing_idx = np.ones((len(strains.keys()), 3))
    
    for region_nr, strain in strains.iteritems():
        for direction in range(3):
            # If all measurements are zero, then it is missing
            if not np.any([s[direction] for s in strain]):
                missing_idx[region_nr-1][direction] = 0
        
            
    return missing_idx






def calculate_fiber_strain(fib, e_circ, e_rad, e_long, strain_markers, mesh, strains):
    
    import dolfin
    from dolfin import Measure, Function, TensorFunctionSpace, VectorFunctionSpace, \
        TrialFunction, TestFunction, inner, assemble_system, solve
    dX = dolfin.Measure("dx", subdomain_data=strain_markers, domain=mesh)

    fiber_space = fib.function_space()
    strain_space = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=3)

    full_strain_space = dolfin.TensorFunctionSpace(mesh, "R", 0)


    fib1 = dolfin.Function(strain_space)
    e_c1 =  dolfin.Function(strain_space)
    e_r1 =  dolfin.Function(strain_space)
    e_l1 =  dolfin.Function(strain_space)

    mean_coords, coords = get_regional_midpoints(strain_markers, mesh)
    # ax = plt.subplot(111, projection='3d')


    region = 1
    fiber_strain = []

    for region in range(1,18):
        # For each region

        # Find the average unit normal in the fiber direction
        u = dolfin.TrialFunction(strain_space)
        v = TestFunction(strain_space)
        a = inner(u, v)*dX(region)
        L_fib = inner(fib, v)*dX(region)
        A, b = assemble_system(a, L_fib)
        solve(A, fib1.vector(), b)
        fib1_norm = np.linalg.norm(fib1.vector().array())
        # Unit normal
        fib1_arr = fib1.vector().array() / fib1_norm

        # Find the average unit normal in Circumferential direction
        u = TrialFunction(strain_space)
        v = TestFunction(strain_space)
        a = inner(u, v)*dX(region)
        L_c = inner(e_circ, v)*dX(region)
        A, b = assemble_system(a, L_c)
        solve(A, e_c1.vector(), b)
        e_c1_norm = np.linalg.norm(e_c1.vector().array())
        # Unit normal
        e_c1_arr = e_c1.vector().array() / e_c1_norm


        # Find the averag unit normal in Radial direction
        u = TrialFunction(strain_space)
        v = TestFunction(strain_space)
        a = inner(u, v)*dX(region)
        L_r = inner(e_rad, v)*dX(region)
        A, b = assemble_system(a, L_r)
        solve(A, e_r1.vector(), b)
        e_r1_norm = np.linalg.norm(e_r1.vector().array())
        # Unit normal
        e_r1_arr = e_r1.vector().array() / e_r1_norm

        # Find the average unit normal in Longitudinal direction
        u = TrialFunction(strain_space)
        v = TestFunction(strain_space)
        a = inner(u, v)*dX(region)
        L_l = inner(e_long, v)*dX(region)
        A, b = assemble_system(a, L_l)
        solve(A, e_l1.vector(), b)
        e_l1_norm = np.linalg.norm(e_l1.vector().array())
        # Unit normal
        e_l1_arr = e_l1.vector().array() / e_l1_norm


        # ax.plot([mean_coords[region][0], mean_coords[region][0]+e_c1_arr[0]],[mean_coords[region][1], mean_coords[region][1]+e_c1_arr[1]], [mean_coords[region][2],mean_coords[region][2]+e_c1_arr[2]], 'b', label = "circ")
        # ax.plot([mean_coords[region][0],mean_coords[region][0]+e_r1_arr[0]],[mean_coords[region][1], mean_coords[region][1]+e_r1_arr[1]], [mean_coords[region][2],mean_coords[region][2]+e_r1_arr[2]] , 'r',label = "rad")
        # ax.plot([mean_coords[region][0],mean_coords[region][0]+e_l1_arr[0]],[mean_coords[region][1], mean_coords[region][1]+e_l1_arr[1]], [mean_coords[region][2],mean_coords[region][2]+e_l1_arr[2]] , 'g',label = "long")
        # ax.plot([mean_coords[region][0],mean_coords[region][0]+fib1_arr[0]],[mean_coords[region][1], mean_coords[region][1]+fib1_arr[1]], [mean_coords[region][2],mean_coords[region][2]+fib1_arr[2]] , 'y', label = "fib")

        fiber_strain_region = []
 
        for strain in strains[region]:

            mat = np.array([strain[0]*e_c1_arr, strain[1]*e_r1_arr, strain[2]*e_l1_arr]).T
            fiber_strain_region.append(np.linalg.norm(np.dot(mat, fib1_arr)))

        fiber_strain.append(fiber_strain_region)

    # for i in range(18):
    #     ax.scatter3D(coords[i][0], coords[i][1], coords[i][2], s = 0.1)

    # plt.show()

    return fiber_strain





def extrapolate_to_final_strain(strains, use_scipy = False):
    """
    In the impact patients the final strain is missing.
    This strain should be = 0, and by estimating it's
    value we can estimate the drift.
    """

    
    
    n = len(strains)
    x = range(n)
    y = strains


    if use_scipy:
        # We should move to this later, since this is established code from scipy. 
        from scipy.interpolate import InterpolatedUnivariateSpline
        # Use second order spline to approximate the next point
        spline = InterpolatedUnivariateSpline(x, y, k=2)
        y_extra = s(n)

    else:
        s = Spline(x, y, 2)
        y_extra = s.extrapolate()
    
   
    return y_extra


class Spline:
    def __init__(self, x,y,p):

        assert len(x) == len(y), "Length of x and y must be the same"

        # Interior knots, free boundary conditions
        # n = len(x)

        X = x[-2*p:]
        
        t_int = [xi for xi in X[p/2 +1:-p/2]]

        # t = [0,0,0,0, 1,1, 2,2, 3,3, 4,4,4,4]

        # Make the knot vector p+1 regular
        self.t = np.concatenate((np.concatenate((X[0]*np.ones(p+1), t_int)), X[-1]*np.ones(p+1)))
        self.x = X
        self.y = y[-2*p:]
        self.p = p
        
        # Coefficients 
        A = self.make_B_spline_matrix(self.p, self.t, self.x)
        self.c = np.linalg.solve(A, self.y)

    def plot_spline(self):
        N = 40
        q = np.zeros((N,1))
        count = 0
        # Spline approximation
        for xi in np.linspace(0,self.x[-1]-0.00001,N):
            mu = self.find_mu(self.t, xi, self.p)
            c0 =  self.c[mu-self.p:mu+1]
            t0 = self.t[mu-self.p+1:mu+self.p+1]
            B = self.algorithm_2_21(mu, self.t, xi, self.p)
            q[count] = np.dot(B,c0)
            count += 1

        plt.figure()
        plt.plot(self.x,self.y, "r*")
        plt.plot(np.linspace(0,self.x[-1],N),q, 'g')
        plt.show()

    def extrapolate(self):
        x_extra = self.x[-1] + 1.0
        
        mu = self.find_mu(self.t, x_extra, self.p, len(self.t)-self.p-2)
        c0 =  self.c[mu-self.p:mu+1]
        t0 = self.t[mu-self.p+1:mu+self.p+1]
        B = self.algorithm_2_21(mu, self.t, x_extra, self.p)
        y_extra = np.dot(B,c0)

        return y_extra


    def plot_extrapolation(self):

        N = 40
        q_extra = np.zeros((N,1))
        count = 0
        for xi in np.linspace(self.x[-3],self.x[-1]+1,N):
            mu = self.find_mu(self.t, xi, self.p, len(self.t)-self.p-2)
            c0 =  self.c[mu-self.p:mu+1]
            t0 = self.t[mu-self.p+1:mu+self.p+1]
            B = self.algorithm_2_21(mu, self.t, xi, self.p)
            q_extra[count] = np.dot(B,c0)
            count += 1

        plt.figure()
        plt.plot(self.x,self.y, "r*")
        plt.plot(self.x[-1]+1, q_extra[-1], 'bx')
        plt.plot(np.linspace(self.x[-3],self.x[-1]+1,N),q_extra, 'g')
        plt.show()
        

    def make_B_spline_matrix(self, p, t, u):
        """ Make Matrix of B splines evaluated at 
        parameter values

        *Arguments*
           p (int)
               polynomial order
           t (array of non-decreasing floats)
               knot vector
           u (array of n floats)
               points of evaluation

        *Return*
           B (n times lent(t) - (p+1) matrix)
               matrix of B splines evaluated 
               at parameter values
        """
        EPS = 1e-8
        # Allocate enough memory 
        A = np.zeros((len(u), len(t)-(p+1)))
        i = 0
        for ui in u:
            # Find the correct knot span
            if ui == t[-1]:
                # Choose a slighty smaller value
                mu = self.find_mu(t, ui-EPS, p)
            else:
                mu = self.find_mu(t, ui, p)

            # Compute the non-zero B-splines evaluted at ui
            Bp = self.algorithm_2_21(mu, t, ui, p)
            # Insert these values into the matrix        
            A[i, mu-p:mu+1] = Bp
            i += 1

        return A

    def find_mu(self,t, x, p, mu = None):
        """
        Return index mu of knot such that
        t[mu] < T < t[mu+1]

        *Arguments*
           x (int)
               value between t[0] and t[n+p+1]
           t (array of non-decreasing floats)
               knot vector
           p (int)
              polynomial order

        *Return*
           mu (int)
              index
        """
        if mu is not None:
            return mu

        # Check that x is contained in some interval
        if x > t[-1]:
            raise ValueError("Value is larger than the largest knot")

        if x < t[0]:
            raise ValueError("Value is smaller than the smallest knot")

        # Flag
        index_ok = False
        # Find the first interval containing x and return the index
        for i in range(len(t)-1):
            if (x >= t[i] and x < t[i+1]):
                mu_false = i
                if i >= p:
                    index_ok = True
                    mu = i
                    break

        # Index is too small. Unable to pick enough control points and knots.
        if not index_ok:
            raise ValueError("Interval found, but index is too small."+ 
            "Try to insert more knots before index {}.".format(mu_false))


        return mu

    def algorithm_2_21(self, mu, t, x, p):
        """
        Calculate B-spline vector given in
        equation 2.21
        (Sparse version)

        *Arguments*
           p (int)
               polynomial order
           t (array of non-decreasing floats)
               knot vector pf size 2*p
           x (float)
               points of evaluation

        *Return*
           B (1 times p+1 matrix)
               vector  given in equation 2.21
        """

        B = np.ones(1)
        for k in range(p):
            R = self.R_matrix(k+1,x, mu, t)
            B_temp = np.ones(k+2)
            B_temp[0] = B[0]*R[0][0]
            for j in range(1,k+1):
                B_temp[j] = B[j-1]*R[j-1][1] + B[j]*R[j][0] 

            B_temp[k+1] = B[k]*R[k][1]

            B = B_temp

        return B

    def R_matrix(self, k, x, mu, t):
        """
        Return matrix given in equation 2.20
        in the compendium
        (Sparse version)

        *Arguments*
           k (int)
               matrix index between 1 <= k <= p
           x (float)
               points of evaluation
           p (int)
               polynomial order
           t (array of non-decreasing floats)
               knot vector pf size 2*p

        *Return*
           R (k times 2 array)
               matrix given in equation 2.20 
               in sparse format
        """

        R = np.zeros((k,2))
        for i in range(k):
            if (t[mu+1+i]- t[mu+1+i-k]) != 0:
                R[i][0] = (t[mu+1+i] - x)/(t[mu+1+i]- t[mu+1+i-k])
                R[i][1] =  (x-t[mu+1+i-k])/(t[mu+1+i]- t[mu+1+i-k])

        return R
