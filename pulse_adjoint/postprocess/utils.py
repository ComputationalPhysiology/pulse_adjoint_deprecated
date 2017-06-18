#!/usr/bin/env python
"""
This script includes different functionlality that is needed
to compute the different features that we want to visualise.
"""
# Copyright (C) 2017 Henrik Finsberg
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
from .args import *

    
def asint(s):
    try: return int(s), ''
    except ValueError: return sys.maxint, s

def get_fiber_field(patient):

    if hasattr(patient, "e_f"):
        e_f = patient.e_f
    else:
        idx_arr = np.where([item.startswith("fiber") for item in dir(patient)])[0]
        
        if len(idx_arr) == 1:
            att = dir(patient)[idx_arr[0]]
            e_f = getattr(patient, att)
            
        else:
            raise ValueError("Unable to find fiber field")

    return e_f
    
    

def init_spaces(mesh, gamma_space = "CG_1"):

    from pulse_adjoint.utils import QuadratureSpace
    
    spaces = {}
    
    spaces["marker_space"] = dolfin.FunctionSpace(mesh, "DG", 0)
    spaces["stress_space"] = dolfin.FunctionSpace(mesh, "DG", 0)
    

    if gamma_space == "regional":
        spaces["gamma_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim = 17)
    else:
        gamma_family, gamma_degree = gamma_space.split("_")
        spaces["gamma_space"] = dolfin.FunctionSpace(mesh, gamma_family, int(gamma_degree))
        
    spaces["displacement_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    spaces["pressure_space"] = dolfin.FunctionSpace(mesh, "CG", 1)
    spaces["state_space"] = spaces["displacement_space"]*spaces["pressure_space"]
    spaces["strain_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=3)
    spaces["strainfield_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 1)

    
    spaces["quad_space"] = QuadratureSpace(mesh, 4, dim = 1)
    
    return spaces

def compute_apical_registration(mesh, patient, endo_surf_apex):
    """
    Compute the displacement between the apex position 
    in the mesh and the apex position in the 
    segmented surfaces
    """

    ffun = patient.ffun
    ENDO = patient.ENDO
 
    endo_facets = np.where(ffun.array() == ENDO)
 
    endo_mesh_apex = [-np.inf, 0, 0]

    for f in dolfin.facets(mesh):
       
        if ffun[f] == ENDO:
            for v in dolfin.vertices(f):
                if v.point().x() > endo_mesh_apex[0]:
                    endo_mesh_apex = [v.point().x(),
                                      v.point().y(),
                                      v.point().z()]
   
    d_endo = np.subtract(endo_surf_apex, endo_mesh_apex)
   
    u = dolfin.Function(dolfin.VectorFunctionSpace(mesh, "CG", 1))
    u.assign(dolfin.Constant([d_endo[0], 0,0]))
    return u
   
   
def get_regional(dx, fun, fun_lst, regions = range(1,18), T_ref=1.0):
    """Return the average value of the function 
    in each segment

    :param dx: Volume measure marked according of AHA segments
    :param fun: The function that should be averaged
    :returns: The average value in each of the 17 AHA segment
    :rtype: list of floats

    """

    if fun.value_size() > 1:
        if len(fun_lst) == 1:
            return T_ref*fun_lst[0]
        else:
            return np.multiply(T_ref, fun_lst)

    # if len(fun.vector()) == 1:
        # return fun.vector().array()[0]*np.ones(len(regions))

    meshvols = []
    for i in regions:
        meshvols.append(dolfin.Constant(dolfin.assemble(dolfin.Constant(1.0)*dx(i))))

    lst = []
    for f in fun_lst:
        fun.vector()[:] = f
        lst_i = [] 
        for t, i in enumerate(regions):
            lst_i.append(T_ref*dolfin.assemble((fun/meshvols[t])*dx(i)))

        lst.append(lst_i)

    if len(fun_lst) == 1:
        return np.array(lst[0])


    return np.array(lst).T

def get_global(dx, fun, fun_lst, regions = range(1,18), T_ref = 1.0):
    """Get average value of function

    :param dx: Volume measure marked according of AHA segments
    :param fun: A dolfin function in the coorect spce
    :param fun_lst: A list of vectors that should be averaged
    :returns: list of average values (one for each element in fun_list)
    :rtype: list of floats

    """
    
    meshvols = []

    for i in regions:
        meshvols.append(dolfin.assemble(dolfin.Constant(1.0)*dx(i)))

    meshvol = np.sum(meshvols)

    fun_mean = []
    for f in fun_lst:
   
        fun.vector()[:] = f
        
        if fun.value_size() > 1:
            fun_tot = np.sum(np.multiply(fun.vector().array(), meshvols))
            
            
        else:
            fun_tot = 0
            for i in regions:            
                fun_tot += dolfin.assemble((fun)*dx(i))

        fun_mean.append(T_ref*fun_tot/meshvol)
 
    return fun_mean

def update_nested_dict(d,u):

    from collections import Mapping
    
    def update(d,u):
        for k, v in u.iteritems():
            if isinstance(v, Mapping):
                r = update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        return d

    update(d,u)

def recompute_strains_to_original_reference(strains, ref):

    strain_dict = {strain : {i:[] for i in STRAIN_REGION_NUMS}  for strain in STRAIN_NUM_TO_KEY.values()}
    
    for d in ['longitudinal', 'circumferential', 'radial']:
        for r in range(1,18):
            strain_trace = strains[d][r]
            new_strain_trace = np.zeros(len(strain_trace))
            ea0 = strain_trace[ref]
        
            for i in range(len(strain_trace)):
                                
                ei0 = strain_trace[i]
                eia = (ei0 - ea0)/(ea0 + 1)
                new_strain_trace[i] = eia

            new_strain_trace = np.roll(new_strain_trace, -ref)
            strain_dict[d][r] = new_strain_trace
            
    return strain_dict
def compute_inner_cavity_volume(mesh, ffun, marker, u=None, approx="project"):
    """
    Compute cavity volume using the divergence theorem. 

    :param mesh: The mesh
    :type mesh: :py:class:`dolfin.Mesh`
    :param ffun: Facet function
    :type ffun: :py:class:`dolfin.MeshFunction`
    :param int endo_lv_marker: The marker of en endocardium
    :param u: Displacement
    :type u: :py:class:`dolfin.Function`
    :returns vol: Volume of inner cavity
    :rtype: float

    """
    dS = dolfin.Measure("exterior_facet", subdomain_data=ffun, domain=mesh)(marker)
    from pulse_adjoint.optimization_targets import VolumeTarget
    target = VolumeTarget(mesh, dS, "LV", approx)
    target.set_target_functions()
    target.assign_simulated(u)
    return target.simulated_fun.vector().array()[0]



def get_volumes(disps, patient, chamber = "lv", approx="project"):


    if chamber == "lv":

        if patient.markers.has_key("ENDO"):
            marker = patient.markers["ENDO"][0]
        elif patient.markers.has_key("ENDO_LV"):
            marker = patient.markers["ENDO_LV"][0]
        else:
            raise ValueError

    else:
        
        assert chamber == "rv"

        if not patient.markers.has_key("ENDO_RV"):
            return []

        marker = patient.markers["ENDO_RV"][0]
        
        
    V = dolfin.VectorFunctionSpace(patient.mesh, "CG", 2)
    u = dolfin.Function(V)
   
 
    ffun = patient.ffun
    
    volumes = []
    if isinstance(disps, dict):
        times = sorted(disps.keys(), key=asint)
    else:
        times = range(len(disps))
    for t in times:
        us = disps[t]
        u.vector()[:] = us

      
        volumes.append(compute_inner_cavity_volume(patient.mesh, ffun,
                                                   marker, u, approx))

   
    return volumes

def get_regional_strains(disps, patient, unload=False,
                         strain_approx = "original",
                         strain_reference="0",
                         strain_tensor="gradu",
                         map_strain = False,
                         *args, **kwargs):


    
    from pulse_adjoint.optimization_targets import RegionalStrainTarget
    dX = dolfin.Measure("dx",
                 subdomain_data = patient.sfun,
                        domain = patient.mesh)


    load_displacemet = (unload and not strain_reference== "unloaded") or \
                       (not unload and strain_reference == "ED")

    
    if load_displacemet:
    
        if strain_reference == "0":
            idx = 1
        else:
            #strain reference =  "ED"
            if unload:
                idx = patient.passive_filling_duration
            else:
                idx =  patient.passive_filling_duration-1

        u0 = dolfin.Function(dolfin.VectorFunctionSpace(patient.mesh,"CG", 2))
        if isinstance(disps, dict):
            u0.vector()[:] = disps[str(idx)]
        else:
            u0.vector()[:] = disps[idx]

        V = dolfin.VectorFunctionSpace(patient.mesh, "CG", 1)
        if strain_approx in ["project","interpolate"]:
            
            if strain_approx == "project":
                u0 = dolfin.project(u0, V)
            else:
                u0 = dolfin.interpolate(u0, V)

        else:
            u_int = dolfin.interpolate(u0, V)
                    
                
        F_ref = dolfin.grad(u0) + dolfin.Identity(3)
                

    else:
        F_ref = dolfin.Identity(3)


    crl_basis = {}
    basis_keys = []
    for att in ["circumferential", "radial", "longitudinal"]:
        if hasattr(patient, att):
            basis_keys.append(att)
            
            crl_basis[att] = getattr(patient, att)
            
    target = RegionalStrainTarget(patient.mesh,
                                  crl_basis, dX,
                                  F_ref =F_ref,
                                  approx = strain_approx,
                                  tensor = strain_tensor,
                                  map_strain=map_strain)
    target.set_target_functions()
   
    
    regions = target.regions

    strain_dict = {}
    
    for d in basis_keys:
        strain_dict[d] = {int(i):[] for i in regions}

    
    
   
    V = dolfin.VectorFunctionSpace(patient.mesh, "CG", 2)
    u = dolfin.Function(V)

    if isinstance(disps, dict):
        times = sorted(disps.keys(), key=asint)
    else:
        times = range(len(disps))
        
    for t in times:
        us = disps[t]
        u.vector()[:] = us

        target.assign_simulated(u)

        for i,d in enumerate(basis_keys):
            for j, r in enumerate(regions):
                strain_dict[d][r].append(target.simulated_fun[j].vector().array()[i])
                

    # error = np.sum([np.subtract(patient.strain[i].T[0],strain_dict["circumferential"][i][1:])**2 for i in range(3)], 0)
    # print error
    # exit()
    return strain_dict

def compute_strain_components(u, sfun, crl_basis, region, F_ref = dolfin.Identity(3), tensor_str="gradu"):

    mesh = sfun.mesh()
    dmu = dolfin.Measure("dx",
                         subdomain_data = sfun,
                         domain = mesh)
 
    # Strain tensor
    I = dolfin.Identity(3)
    F = (dolfin.grad(u) + dolfin.Identity(3))*dolfin.inv(F_ref)
    
    if tensor_str == "gradu":
        tensor = F-I
    else:
        C = F.T * F
        tensor = 0.5*(C-I)

    # Volume of region
    vol = dolfin.assemble(dolfin.Constant(1.0)*dmu(region))

    # Strain components
    return [dolfin.assemble(dolfin.inner(e,tensor*e)*dmu(region))/vol for e in crl_basis]

def interpolate_arr(x, arr, N, period = None, normalize = True):

    # from scipy.interpolate import splrep
    a_min = np.min(arr)
    a_max = np.max(arr)
    x_min = np.min(x)
    x_max = np.max(x)

    if a_min == a_max:
        # The array is constant
        return a_min*np.ones(N)
        
    # x = np.linspace(0,1,len(arr))

    # Normalize
    if normalize:
        arr = np.subtract(arr,a_min)
        arr = np.divide(arr, a_max-a_min)
        x = np.subtract(x, x_min)
        x = np.divide(x, x_max-x_min)

    # Interpolate
    xp = np.linspace(0,1,N)
    # try:
    fp = np.interp(xp,x,arr, period=period)
    # except:

    
    fp = np.multiply(fp, a_max-a_min)
    fp = np.add(fp, a_min)

    return fp

def interpolate_trace_to_valve_times(arr, valve_times, N):
    """
    Given an array and valvular timings, perform interpolation
    so that the resulting array is splitted up into chunks of length
    N, each chuck being the interpolated values of the array for 
    one valvular event to the next. 

    First list is from 'mvc' to 'avo'.
    Second list is from 'avo' to 'avc'.
    Third list is from 'avc' to 'mvo'.
    Fourth list is from 'mvo' to 'mvc'.

    :param arr: The array of interest 
    :type arr: `numpy.array` or list 
    :param dict valve_times: A dictionary of valvular timings
    :param int N: length of chunks
    :returns: a list of length 4, with each element being a
              list of length N. 
    :rtype: list

    """
    

    echo_valve_times = valve_times
    # The index when the given array is starting
    pfb = valve_times["passive_filling_begins"]
    
    n = len(arr)
    # Just some increasing sequence
    time = np.linspace(0,1, len(arr))
    

    # Roll array so that it start on the same index and in the valvular times
    arr_shift_pdb = np.roll(arr, pfb)
    # gamma_mean_shift = gamma_mean

    full_arr = []

    N_ = {"avo": int(3*N*float(0.05)), "avc":int(3*N*float(0.35)),
          "mvo":int(3*N*float(0.10)), "end":int(3*N*float(0.50)) }
    
    for start, end in [("mvc", "avo"), ("avo", "avc"), ("avc", "mvo"), ("mvo", "end")]:
        
        start_idx = echo_valve_times[start]
        end_idx = echo_valve_times[end]
        diff = (end_idx - start_idx) % n

        # If start and end are the same, include the previous point
        if diff == 0:
            start_idx -= 1
            diff = 1
        # if end == "mvc":
        #     diff -= 0

        # Roll array to this start
        arr_shift_start = np.roll(arr_shift_pdb, -start_idx)
        arr_partly = arr_shift_start[:diff+1]
        
        # The time starts at mvc
        time_shift_start = np.roll(arr_shift_pdb, echo_valve_times["mvc"]-start_idx)
        t_partly = time_shift_start[:diff+1]

        # just some increasing sequence
        dtime = time[:diff+1]
                
            
        darr_int = interpolate_arr(dtime, arr_partly, N_[end])
   
        
        full_arr.append(darr_int)
 
    return np.concatenate(full_arr)
def compute_elastance(state, pressure, gamma, patient,
                      params, matparams, return_v0 = False):
    """FIXME! briefly describe function

    :param state: 
    :param pressure: 
    :param gamma: 
    :param gamma_space_str: 
    :param patient: 
    :param active_model: 
    :param matparams: 
    :param return_v0: 
    :returns: 
    :rtype: 

    """
    

    solver, p_lv = get_calibrated_solver(state, pressure,
                                         gamma, patient,
                                         params,matparams)

    p_lv.t = pressure
    solver.solve()
   
    u,_ = dolfin.split(solver.get_state())
    volume = compute_inner_cavity_volume(patient.mesh, patient.ffun,
                                         patient.markers["ENDO"][0], u)

    vs = [volume]
    ps = [pressure]

    print "Original"
    print "{:10}\t{:10}".format("pressure", "volume")
    print "{:10.2f}\t{:10.2f}".format(pressure, volume)
    print "Increase the pressure"

    n = 1
    inc = 0.1
    crash = True
    while crash:
        # Increase the pressure
        p_lv.t += inc
        # Do a new solve
        try:
            solver.solve()
        except SolverDidNotConverge:
            inc /= 2.0
            continue

        else:
            # Compute the new volume
            u,_ = dolfin.split(solver.get_state())
            v = compute_inner_cavity_volume(patient.mesh, patient.ffun,
                                            patient.markers["ENDO"][0], u)
            
            print "{:10.2f}\t{:10.2f}".format(p_lv.t, v)
            # Append to the list
            vs.append(v)
            ps.append(p_lv.t)

            crash = False

    if return_v0:
        e = np.mean(np.divide(np.diff(ps), np.diff(vs)))
        v0 = volume - float(pressure)/e
        return e, v0
    else:
        return np.mean(np.divide(np.diff(ps), np.diff(vs)))


def compute_geometric_distance(patient, us, vtk_output):
    """Compute the distance between the vertices from the simulation
    and the vertices from the segmented surfaces of the endocardium.
    For each vertex in the simulated surface :math:`a \in \Xi_{\mathrm{sim}}`,
    we define the following distance measure

    .. math::

       d(a,\Xi_{\mathrm{seg}}) = \min_{b \in \Xi_{\mathrm{seg}}} \| a - b \| 

    where 

    .. math:: 

       \Xi_{\mathrm{seg}} 

    is the vertices of the (refined) segmented surface

    :param patient: Patient class
    :param us: list of displacemets
    :param vtk_output: directory were to save the output
    :returns: 
    :rtype: 

    """
    
    import vtk_utils
    import vtk
    
    
    V_cg1 = dolfin.VectorFunctionSpace(patient.mesh, "CG", 1)
    V_cg2 = dolfin.VectorFunctionSpace(patient.mesh, "CG", 2)
    u_current = dolfin.Function(V_cg2)
    u_prev = dolfin.Function(V_cg2)
    d = dolfin.Function(V_cg2)

    mean_dist = []
    max_dist = []
    std_dist = []
   

    for k,t in enumerate(np.roll(range(patient.num_points), -patient.passive_filling_begins)):

        mesh = patient.mesh

        if not us.has_key(str(k)):
            print("Time point {} does not exist".format(k))
            continue
        u_current.vector()[:] = us[str(k)]
        d.vector()[:] =  u_current.vector()[:] - u_prev.vector()[:]
        ud = dolfin.interpolate(d, V_cg1)
        dolfin.ALE.move(mesh, ud)
       
        
        endoname = vtk_utils.save_surface_to_dolfinxml(patient,t, vtk_output)
        endo_surf = dolfin.Mesh(endoname)
        endo_surf_apex = endo_surf.coordinates().T[0].max()
            
        # Registrer the apex
        u_apical = compute_apical_registration(mesh, patient, endo_surf_apex)
        dolfin.ALE.move(mesh, u_apical)
        

        # Save unrefined surface for later visualization
        surf_unrefined = vtk_utils.dolfin2polydata(endo_surf)
        distname = "/".join([vtk_output, "echopac_{}.vtk".format(k)])
        vtk_utils.write_to_polydata(distname, surf_unrefined)

        # Convert surface to dolfin format
        # surf_unrefined = dolfin2vtu(endo_surf)

        # Refine surface for better accuracy
        endo_surf_refined = dolfin.refine(dolfin.refine(dolfin.refine(dolfin.refine(endo_surf))))
        # Get endocardial mesh from original mesh
        endo_submesh = vtk_utils.get_submesh(mesh, patient.ENDO)

        # Convert surfaces to polydata
        endo_surf_vtk = vtk_utils.dolfin2polydata(endo_surf_refined)
        endo_submesh_vtk = vtk_utils.dolfin2polydata(endo_submesh)
        
        # Build a Kd search tree 
        tree = vtk.vtkKdTreePointLocator()
        tree.SetDataSet(endo_surf_vtk)
        tree.BuildLocator()

        distance = vtk.vtkDoubleArray()
        distance_arr = []
        for i in range(endo_submesh_vtk.GetNumberOfPoints()):
            p = endo_submesh_vtk.GetPoint(i)

            # Nearest neighbor
            idx = tree.FindClosestPoint(p)
            psurf = endo_surf_vtk.GetPoint(idx)

            # Compute di
            dist = np.linalg.norm(np.subtract(psurf,p))
            
            distance.InsertNextValue(dist)
            distance_arr.append(dist)

        # Set the distances as scalars in the vtk file
        endo_submesh_vtk.GetPointData().SetScalars(distance)

        distname = "/".join([vtk_output, "dist_{}.vtk".format(k)])
        vtk_utils.write_to_polydata(distname, endo_submesh_vtk)

        mean_dist.append(np.mean(distance_arr))
        std_dist.append(np.std(distance_arr))
        max_dist.append(np.max(distance_arr))

        u_prev.assign(u_current)

    d = {"mean_distance": mean_dist,
         "std_distance": std_dist,
         "max_distance": max_dist}
    return d

def get_Ivol(simulated, measured):
    """
    return the relatve error in l1 norm
    || V* - V ||_l1 / || V ||_l1 where V* is
    simulated volume and V is measured volume
    """
    if not len(simulated) == len(measured):
        print("All simulation points are not available")
        n = len(simulated)
        measured = measured[:n]
        
    return np.sum(np.abs(np.subtract(simulated,measured))) / \
        float(np.sum(measured))

def get_Istrain(simulated,measured):
    """
    Return two different measures for the strain error
    
    max:
    ||e* - e ||
    """
    I_strain_tot_rel = 0
    I_strain_tot_max = 0
    for d in measured.keys():
        
        I_strain_region_rel = []
        I_strain_region_max = []
        
        s_max = np.max([np.max(np.abs(s)) for s in measured[d].values()])
        for region in measured[d].keys():
            
            s_meas = measured[d][region]
            s_sim =  simulated[d][region]
            
            if not np.all(s_meas == 0):

                if not len(s_meas) == len(s_sim):
                    print("All simulation points are not available")
                    n = len(s_sim)
                    s_meas = s_meas[:n]
                    
                err_max =  np.divide(np.mean(np.abs(np.subtract(s_sim,s_meas))),
                                    s_max)
                err_rel = np.divide(np.sum(np.abs(np.subtract(s_sim,s_meas))),
                                    np.sum(np.abs(s_meas)))
                
                I_strain_region_max.append(err_max)
                I_strain_region_rel.append(err_rel)
  
        I_strain_tot_rel += np.mean(I_strain_region_rel)
        I_strain_tot_max += np.mean(I_strain_region_max)
                
    I_strain_rel = I_strain_tot_rel/3.
    I_strain_max = I_strain_tot_max/3.

    return I_strain_rel, I_strain_max

def copmute_data_mismatch(us, patient, measured_volumes, measured_strains):

    simulated_volumes = get_volumes(us, patient)
    simulated_strains = get_regional_strains(us, patient)
        
    I_vol = get_Ivol(simulated_volumes, measured_volumes)
    I_strain_rel, I_strain_max = get_Istrain(simulated_strains,
                                             measured_strains)

    data = {"I_strain_rel": I_strain_rel,
            "I_strain_max": I_strain_max,
            "I_vol": I_vol}

    return data

def compute_time_varying_elastance(patient, params, data):
    """Compute the elastance for every point in
    the cycle.

    :param patient: Patient class
    :param matparams: Optimal material parameters
    :param params: pulse_adjoint.adjoint_contraction_parameters
    :param val: data
    :returns: time varying elastance
    :rtype: list

    """

    
    matparams = {k:v[0] for k,v in data["material_parameters"].iteritems()}
    
    elastance = []
    dead_volume = []

    num_points = patient.num_points
    if params["unload"]: num_points += 1
    start = 1 if params["unload"] else 0
    
    for i in range(start, num_points):
        print "{} / {} ".format(i, num_points)
        
        p = patient.pressure[i]
        w = data["states"][str(i)]
        g = data["gammas"][str(i)]
        
        e, v0 = compute_elastance(w, p, g, patient, params,
                                  matparams, return_v0 = True)
        
        print "E = {}, V0 = {}".format(e, v0)
        elastance.append(e)
        dead_volume.append(v0)

    d = {"elastance": elastance, "v0":dead_volume}
    return d
    
    


def compute_cardiac_work_echo(stresses, strains, flip =False):
    """FIXME! briefly describe function

    :param list stresses: list of stresses
    :param list strains: list of strains
    :param bool flip: If true, change the sign on the stresses.
                      This is done in the original paper, when the
                      pressure plays the role as stress.
    :returns: the work
    :rtype: list

    """
    

    msg =  "Stresses and strains do not have same lenght"
    assert len(stresses) == len(strains), msg

    # Compute the averge
    stress_avg = np.add(stresses[:-1], stresses[1:])/2.0
    
    if flip:
        # Compute the shortening_rate
        dstrain = -np.diff(strains)
    else:
        # Compute the strain_rate
        dstrain = np.diff(strains)

    # The work is the cumulative sum of the product
    work = np.append(0,np.cumsum(dstrain*stress_avg))
    
    return work
    
    
    

def compute_cardiac_work(patient, params, val, case, wp):
    """Compute cardiac work. 

    :param patient: patient data
    :param params: pulse_adjoin.adjoint_contraction_parameters
    :param val: the data
    :param path: path to where to save the output

    """
    
    from cardiac_work import CardiacWork, CardiacWorkEcho, StrainEnergy


    spaces = get_feature_spaces(patient.mesh, params["gamma_space"])

    pressures = patient.pressure
    matparams = {k:v[0] for k,v in val["material_parameters"].iteritems()}

    states = val["states"]
    gammas = val["gammas"]
    times = sorted(states.keys(), key=asint)

    if params["unload"]:
        times = times[1:]
    

    dX = dolfin.Measure("dx",subdomain_data = patient.sfun,
                        domain = patient.mesh)
    
    V = dolfin.TensorFunctionSpace(patient.mesh, "DG", 1)
    W = dolfin.FunctionSpace(patient.mesh, "DG", 1)
    e_f = get_fiber_field(patient)
    e_l = patient.longitudinal

    
    assert case in cases, "Unknown case {}".format(case)
    assert wp in work_pairs, "Unknown work pair {}".format(wp)

    reults = {}

    header = ("\nComputing Cardiac Work, W = {}\n"
              "{}, region = {}\n")

   
    if wp == "pgradu":
        cw = CardiacWorkEcho(V, W)
    elif wp == "strain_energy":
        cw = StrainEnergy()
    else:
        cw = CardiacWork(V, W)
            

    case_split = case.split("_")
    if len(case_split) == 1:
        e_k = None
        
    elif case_split[1] == "fiber":
        e_k = e_f
        
    else:
        e_k = e_l
        
                
    case_ = case_split[0]
        
    results = {}

    

    cw.reset()

    regions = set(patient.sfun.array())
    work_lst = {r:[] for r in regions}
    power_lst = {r:[] for r in regions}

    # print(header.format(wp, case, region))

    first_time = True
    
    for t in times:

        print "\nTime: {}".format(t)
        state = states[t]
        gamma = gammas[t]
        pressure = pressures[int(t)]
        
        solver, p_lv = get_calibrated_solver(state, pressure,
                                             gamma,
                                             patient,
                                             params, 
                                             matparams)
        
        u,_ = solver.get_state().split(deepcopy=True)
        
        post = solver.postprocess()
            
        # Second Piola stress
        S = -post.second_piola_stress()
        # Green-Lagrange strain
        E = post.GreenLagrange()
        
        # # First Piola stress
        # P = solver.postprocess().first_piola_stress()
        # # Deformation gradient
        # F = post.deformation_gradient()
        
        # Strain energy
        psi = solver.postprocess().strain_energy()
        
        gradu = dolfin.grad(u)
        
        if wp == "strain_energy":
            
            cw(psi, dx)
            
        else:
            if wp == "SE":
                stress = S
                strain = E
            # elif wp == "PF":
            #     stress = P< 
            #     strain = F
            else:# wp == pgradu
                stress = pressure
                strain = gradu
                
                



        
        
        cw(strain, stress, case_, e_k)

        if first_time:
            first_time = False
            continue

        for region in regions:
            dx = dX if region == 0 else dX(int(region))
            meshvol = dolfin.assemble(dolfin.Constant(1.0)*dx)

            power = cw.get_power()
            work = cw.get_work()

            power_ = dolfin.assemble( power * dx ) / meshvol
            work_ = dolfin.assemble( work * dx ) / meshvol

            work_lst[region].append(work_)
            power_lst[region].append(power_)

            print("\t{:<10}\t{:<10.3f}\t{:<10.3f}".format(region, power_, work_))
        

    for region in regions:    
        results["{}_{}_region_{}".format(wp, case, region)] =  {"power":power_lst[region],
                                                                "work":work_lst[region]}
      

    return results
        
    
def get_feature_spaces(mesh, gamma_space = "CG_1"):

    spaces = {}

    spaces["marker_space"] = dolfin.FunctionSpace(mesh, "DG", 0)
    spaces["stress_space"] = dolfin.FunctionSpace(mesh, "DG", 0)
    

    if gamma_space == "regional":
        spaces["gamma_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim = 17)
    else:
        gamma_family, gamma_degree = gamma_space.split("_")
        spaces["gamma_space"] = dolfin.FunctionSpace(mesh, gamma_family, int(gamma_degree))
        
    spaces["displacement_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    spaces["pressure_space"] = dolfin.FunctionSpace(mesh, "CG", 1)
    spaces["state_space"] = spaces["displacement_space"]*spaces["pressure_space"]
    spaces["strain_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=3)
    spaces["strainfield_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 1)

    from pulse_adjoint.utils import QuadratureSpace
    spaces["quad_space"] = QuadratureSpace(mesh, 4, dim = 3)
    spaces["quad_space_1"] = QuadratureSpace(mesh, 4, dim = 1)
    

    return spaces


   
def make_simulation(params, features, outdir, patient):

   

    if not features: return

    import vtk_utils


    # Mesh
    mesh = patient.mesh

    if 0:
        name = params["Patient_parameters"]["patient"]
        fname = "../DATA2/transformation/{}.txt".format(name)
        F = np.loadtxt(fname)

    else:
        F = np.eye(4)
        
    # Mesh that we move
    moving_mesh = dolfin.Mesh(mesh)

    # The time stamps
    if isinstance(features["gamma"], dict):
        times = sorted(features["gamma"].keys(), key=asint)
    else:
        times = range(len(features["gamma"]))

    if not hasattr(patient, "time"):
        patient.time = range(patient.num_points)
        
    time_stamps = np.roll(patient.time, -np.argmin(patient.time))
    from scipy.interpolate import InterpolatedUnivariateSpline
    s = InterpolatedUnivariateSpline(range(len(time_stamps)), time_stamps, k = 1)
    time_stamps = s(np.array(times, dtype=float))
    
    # Create function spaces
    spaces = get_feature_spaces(mesh, params["gamma_space"])
    moving_spaces = get_feature_spaces(moving_mesh, params["gamma_space"])
    if params["gamma_space"] == "regional":
        gamma_space = dolfin.FunctionSpace(moving_mesh, "DG", 0)
        rg = RegionalParameter(patient.sfun)
    else:
        gamma_space = moving_spaces["gamma_space"]

    # Create functions

    # Markers
    sm = dolfin.Function(moving_spaces["marker_space"], name = "AHA-zones")
    sm.vector()[:] = patient.sfun.array()

    if hasattr(params["Material_parameters"]["a"], "vector"):
        matvec = params["Material_parameters"]["a"].vector()
    else:
        matvec = params["Material_parameters"]["a"]
    # Material parameter
    if params["matparams_space"] == "regional":
        mat_space = dolfin.FunctionSpace(moving_mesh, "DG", 0)
        rmat = RegionalParameter(patient.sfun)
        rmat.vector()[:] = matvec
        mat = dolfin.Function(mat_space, name = "material_parameter_a")
        m =  dolfin.project(rmat.get_function(), mat_space)
        mat.vector()[:] = m.vector()
        
    else:
        family, degree = params["matparams_space"].split("_")
        mat_space= dolfin.FunctionSpace(moving_mesh, family, int(degree))
        mat = dolfin.Function(mat_space, name = "material_parameter_a")
        mat.vector()[:] = matvec


    functions = {}
    for f in features.keys():

        if f == "displacement":
            pass
        elif f == "gamma":
            functions[f] = dolfin.Function(gamma_space, name="gamma")
        else:
            functions[f] = dolfin.Function(moving_spaces["stress_space"], 
                                          name=f)


    # Setup moving mesh
    u = dolfin.Function(spaces["displacement_space"])
    u_prev = dolfin.Function(spaces["displacement_space"])
    u_diff = dolfin.Function(spaces["displacement_space"])
    # Space for interpolation
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    fiber = dolfin.Function(moving_spaces["quad_space"])
   
   
    fname = "simulation_{}.vtu"
    vtu_path = "/".join([outdir, fname])

    old_coords = np.ones((moving_mesh.coordinates().shape[0], 4))
    old_coords[:,:3] = moving_mesh.coordinates()
    
    for i,t in enumerate(times):

        moving_mesh.coordinates()[:] = old_coords[:,:3]
        
        u.vector()[:] = features["displacement"][t]
        
        u_diff.vector()[:] = u.vector() - u_prev.vector()
        d = dolfin.interpolate(u_diff, V)
        dolfin.ALE.move(moving_mesh, d)

        
        old_coords = np.ones((moving_mesh.coordinates().shape[0], 4))
        old_coords[:,:3] = moving_mesh.coordinates()
        
        new_coords = np.linalg.inv(F).dot(old_coords.T).T
        moving_mesh.coordinates()[:] = new_coords[:,:3]

        for f in functions.keys():

            if f == "gamma":
        
                if params["gamma_space"] == "regional":
                    rg.vector()[:] = features["gamma"][t]
                    g = dolfin.project(rg.get_function(), gamma_space)
                    functions[f].vector()[:] = g.vector()
                else:
                    functions[f].vector()[:] = features["gamma"][t]
            else:
                functions[f].vector()[:] = features[f][t]


        vtk_utils.add_stuff(moving_mesh, vtu_path.format(i), sm,mat,
                            *functions.values())
        
        u_prev.assign(u)
        

    pvd_path = "/".join([outdir, "simulation.pvd"])
    print "Simulation saved at {}".format(pvd_path)
    vtk_utils.write_pvd(pvd_path, fname, time_stamps[:i+1])


def save_displacements(params, features, outdir):

    from ..patient_data import FullPatient
    import vtk_utils
    
    patient = FullPatient(**params["Patient_parameters"])

    # Mesh
    mesh = patient.mesh

    spaces = get_feature_spaces(mesh, params["gamma_space"])
    u = dolfin.Function(spaces["displacement_space"])

    path = "/".join([outdir, "displacement.xdmf"])
    f = dolfin.XDMFFile(dolfin.mpi_comm_world(), path)
    times = sorted(features["displacement"].keys(), key=asint)

    for i,t in enumerate(times):

        u.vector()[:] = features["displacement"][t]

        f.write(u, float(t))
        
    
def mmhg2kpa(mmhg):
    """Convert pressure from mmgh to kpa
    """
    return mmhg*101.325/760

def kpa2mmhg(kpa):
    """Convert pressure from kpa to mmhg
    """
    return kpa*760/101.325
def compute_emax(patient, params, val, valve_times):
    
    echo_valve_times  = valve_times#["echo_valve_time"]
              
    pfb = patient.passive_filling_begins
    n = patient.num_points
    es_idx = (echo_valve_times["avc"] - pfb) % n

    matparams = {k:v[0] for k,v in val["material_parameters"].iteritems()}
    
    if val["states"].has_key(str(es_idx)):
        p_es = patient.pressure[es_idx]
        w_es = val["states"][str(es_idx)]
        g_es = val["gammas"][str(es_idx)]
        
        print "es_idx = ", es_idx
        return compute_elastance(w_es, p_es, g_es,
                                 patient,
                                 params,
                                 matparams)
    else:
        return None



def copmute_mechanical_features(patient, params, val, path):
    """Compute mechanical features such as stress, strain, 
    works etc, save the output in dolfin vectors to a file, and 
    return a dictionary with average scalar values.

    :param patient: patient data
    :param params: pulse_adjoin.adjoint_contraction_parameters
    :param val: the data
    :param path: path to where to save the output

    """
    
    

    outdir = os.path.dirname(path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print path

    spaces = get_feature_spaces(patient.mesh, params["gamma_space"])

    dx = dolfin.Measure("dx",subdomain_data = patient.sfun,
                        domain = patient.mesh)
    regions = [int(r) for r in set(patient.sfun.array())]

    meshvols = {"global": float(dolfin.assemble(dolfin.Constant(1.0)*dx))}
    for i in regions:
        meshvols[i] = float(dolfin.assemble(dolfin.Constant(1.0)*dx(i)))

    
    pressures = patient.pressure
    rv_pressures = None if not hasattr(patient, "RVP") else patient.rv_pressure

    
    ed_point = str(patient.passive_filling_duration) if params["unload"]\
               else str(patient.passive_filling_duration-1)

    # Material parameter
    matparams = {}
    for k,v in val["material_parameters"].iteritems():
        if np.isscalar(v):
            matparams[k] = v
        else:
            if len(v) == 1:
                matparams[k] = v[0]
            else:
                if params["matparams_space"] == "regional":
                    assert len(v) == len(regions)
                    par = RegionalParameter(patient.sfun)
                    par.vector()[:] = v
                    matparams[k] = par.get_function()

                else:
                    family, degree =  params["matparams_space"].split("_")
                    V = dolfin.FunctionSpace(patient.mesh, family, int(degree))
                    par = dolfin.Function(V)
                    assert len(v) == len(par.vector())
                    par.vector()[:] = v
                    matparams[k] = par
                
    
    states = val["states"]
    gammas = val["gammas"]
    times = sorted(states.keys(), key=asint)

    features = {"green_fiber_strain": [],
                # "deform_fiber_strain": [],
                # "gradu_fiber_strain": [],
                "green_longitudinal_strain": [],
                "green_circumferential_strain_ed": [],
                # "gradu_longitudinal_strain": [],
                # "deform_longitudinal_strain": [],
                "caucy_fiber_stress": [],
                "caucy_fiber_stress_dev": [],
                # "piola1_fiber_stress":[],
                # "piola2_fiber_stress": [],
                "caucy_longitudinal_stress": [],
                # "piola1_longitudinal_stress": [],
                # "piola2_longitudinal_stress": [],
                "gamma": [],
                "displacement": []}

    features_scalar = {"green_fiber_strain":{str(r):[] for r in ["global"]+regions },
                       # "gradu_fiber_strain":{str(r):[] for r in ["global"]+regions },
                       # "deform_fiber_strain": {str(r):[] for r in ["global"]+regions },
                       "green_longitudinal_strain":{str(r):[] for r in ["global"]+regions },
                       "green_circumferential_strain_ed":{str(r):[] for r in ["global"]+regions },
                       # "deform_longitudinal_strain":{str(r):[] for r in ["global"]+regions },
                       # "gradu_longitudinal_strain":{str(r):[] for r in ["global"]+regions },
                       "caucy_fiber_stress":{str(r):[] for r in ["global"]+regions },
                       "caucy_fiber_stress_dev":{str(r):[] for r in ["global"]+regions },
                       # "piola1_fiber_stress":{str(r):[] for r in ["global"]+regions },
                       # "piola2_fiber_stress":{str(r):[] for r in ["global"]+regions },
                       "caucy_longitudinal_stress": {str(r):[] for r in ["global"]+regions },
                       # "piola1_longitudinal_stress":{str(r):[] for r in ["global"]+regions },
                       # "piola2_longitudinal_stress":{str(r):[] for r in ["global"]+regions },
                       "gamma": {str(r):[] for r in ["global"]+regions }}

    print("Extracting the following features:")
    print(features.keys())

    if hasattr(patient, "longitudinal"):
        e_long = patient.longitudinal
        has_longitudinal = True
    else:
        has_longitudinal = False

    if hasattr(patient, "circumferential"):
        e_circ = patient.circumferential
        has_circumferential = True
    else:
        has_circumferential = False

        
        
    e_f = get_fiber_field(patient)
    def get(feature, fun, space, project = True):

        assert space in spaces.keys(), "Invalid space: {}".format(space)
        assert feature in features.keys(), "Invalid feature: {}".format(feature)

        if project:
            f_ = localproject(fun, spaces["quad_space_1"])
            remove_extreme_outliers(f_, 500.0, -100.0)
            f = smooth_from_points(spaces[space], f_)
        else:
            f = fun
            
        features[feature].append(dolfin.Vector(f.vector()))

        if feature != "displacement":

            arr = f.vector().array()            
            regional = get_regional(dx, f, [arr], regions)
            scalar = get_global(dx, f, [arr], regions)
            
            for i,r in enumerate(regions):
                features_scalar[feature][str(r)].append(regional[i])
                
            features_scalar[feature]["global"].append(scalar[0])
        
        
    
    for t in times:

        print("\tTimepoint {}/{} ".format(t, len(times)-1))
        state = states[t]
        gamma = gammas[t]
        pressure = pressures[int(t)]
        rv_pressure = None if rv_pressures is None else rv_pressures[int(t)]
        
        solver, p_lv = get_calibrated_solver(state, pressure,
                                             gamma,
                                             patient,
                                             params, 
                                             matparams, rv_pressure)

        u,p = solver.get_state().split(deepcopy=True)

        post = solver.postprocess()

        w_ed = dolfin.Function(solver.get_state().function_space())
        w_ed.vector()[:] = states[ed_point]
        u_ed, _ = w_ed.split(deepcopy=True)
        F_ed = dolfin.Identity(3) + dolfin.grad(u_ed)
        
        F1 =  (dolfin.Identity(3) + dolfin.grad(u))*dolfin.inv(F_ed)
        E = 0.5*(F1.T*F1 - dolfin.Identity(3))
        Ec = dolfin.inner(E*e_circ, e_circ)

        get("green_fiber_strain", post.green_strain_component(e_f), "stress_space")
        # get("deform_fiber_strain", post.deformation_gradient_component(e_f), "stress_space")
        # get("gradu_fiber_strain", post.gradu_component(e_f), "stress_space")

        get("caucy_fiber_stress", post.cauchy_stress_component(e_f), "stress_space")
        # get("caucy_fiber_stress_dev", post.cauchy_stress_component(e_f, deviatoric=True), "stress_space")
        # get("piola2_fiber_stress", post.piola2_stress_component(e_f), "stress_space")

        if has_longitudinal:
            get("green_circumferential_strain_ed", Ec, "stress_space")
            get("green_longitudinal_strain", post.green_strain_component(e_long), "stress_space")
            # get("deform_longitudinal_strain", post.green_strain_component(e_long), "stress_space")
            # get("gradu_longitudinal_strain", post.gradu_component(e_long), "stress_space")
            get("caucy_longitudinal_stress", post.cauchy_stress_component(e_long), "stress_space")
            # get("caucy_longitudinal_stress", post.cauchy_stress_component(e_long), "stress_space")
            # get("piola1_longitudinal_stress", post.piola1_stress_component(e_long), "stress_space")
            # get("piola2_longitudinal_stress", post.piola2_stress_component(e_long), "stress_space")
            
        get("displacement", u, "displacement_space", False)
        
        gamma = solver.get_gamma()
        gamma.vector()[:] = np.multiply(params["T_ref"], gamma.vector().array())
        get("gamma", gamma, "gamma_space", False)

        
    from load import save_dict_to_h5
    save_dict_to_h5(features, path)

    return features_scalar


def get_solver(matparams, patient, gamma, params):

    from ..setup_optimization import make_solver_parameters
    from ..lvsolver import LVSolver

    solver_parameters, pressure, paramvec= make_solver_parameters(params, patient,
                                                                  matparams, gamma)
    return LVSolver(solver_parameters), pressure


def get_calibrated_solver(state_arr, pressure, gamma_arr,
                          patient, params, matparams, rv_pressure = None):

    
    if params["gamma_space"] == "regional":
        gamma = RegionalParameter(patient.sfun)
        gamma_tmp = RegionalParameter(patient.sfun)
    else:
        gamma_space = dolfin.FunctionSpace(patient.mesh, "CG", 1)
        gamma_tmp = dolfin.Function(gamma_space, name = "Contraction Parameter (tmp)")
        gamma = dolfin.Function(gamma_space, name = "Contraction Parameter")

    solver, p_expr = get_solver(matparams, patient, gamma, params)

    
    gamma_tmp.vector()[:] = gamma_arr
    gamma.assign(gamma_tmp)

    p_lv = p_expr["p_lv"]
    

    w = dolfin.Function(solver.get_state_space())
    w.vector()[:] = state_arr

    solver.reinit(w)

    
    return solver, p_lv


def remove_extreme_outliers(fun, ub=np.inf, lb=-np.inf):
    """
    Set all values that are larger than ub to ub, 
    and set all values that are lower than lb to lb.

    fun : dolfin.Function
        The function from which you want to remove extreme outliers
    ub : float
        Upper bound
    lb : float
        Lower bound
    
    """
    fun.vector()[fun.vector().array() > ub] = ub
    fun.vector()[fun.vector().array() < lb] = lb

    

def smooth_from_points(V, f0, nsamples = 20) :
    """
    Smooth f0 by interpolating f0 into V by using radial basis functions
    for interpolating scattered data using nsamples.
    The higher values of nsamples, the more you smooth.
    
    This is very useful is e.g f0 is a function in a
    quadrature space

    Parameters
    ----------

    V : dolfin.FunctionSpace
        The space for the function to be returned
    f0 : dolfin.Function
        The function you want to smooth
    nsamples : int (optional)
        For each degree of freedom in V, use nsamples to
        build the radial basis function. Default = 20.

    Returns
    -------
    f : dolfin.Function
        The function f0 interpolated into V
        using radial basis funcitions for interpolating 
        scattered data

    """

    from scipy.spatial import cKDTree
    from scipy.interpolate import Rbf
    import numpy as np
    # points for f0
    V0 = f0.function_space()
    # xyz = V0.dofmap().tabulate_all_coordinates(V0.mesh()).reshape(-1, 3)
    xyz =  V0.tabulate_dof_coordinates().reshape((-1, 3))
    f0val = f0.vector().array()
    tree = cKDTree(xyz)

    # coordinate of the dofs
    # coords = V.dofmap().tabulate_all_coordinates(V.mesh()).reshape(-1, 3)
    coords =  V.tabulate_dof_coordinates().reshape((-1, 3))
    f = dolfin.Function(V)

    
    for idx in xrange(0, V.dim()) :
        v = coords[idx,:]
        samples_rad, samples_idx = tree.query(v, nsamples)
        a =  xyz[samples_idx,:]
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, inds = np.unique(b, return_index=True)
        c = a[inds]
        s_idx=samples_idx[inds]

        
        xx, yy, zz = np.split(c, 3, axis=1)
        fvals = f0val[s_idx]
        rbf = Rbf(xx, yy, zz, fvals, function='linear')
        f.vector()[idx] = float(rbf(v[0], v[1], v[2]))
        
        
    return f



def localproject(fun, V) :
    """
    Cheaper way of projecting than regular projections.
    This is useful if you have many degrees of freedom.
    For more info, see dolfin.LocalSolver.

    Parameters
    ---------- 
    fun : dolfin.Function
        The function you want to project
    V : dolfin.FunctionSpace
        The you want to project into
    
    Returns
    -------
    res : dolfin.Function
        fun projected into V
    
    """
    a = dolfin.inner(dolfin.TestFunction(V), dolfin.TrialFunction(V)) * dolfin.dx
    L = dolfin.inner(dolfin.TestFunction(V), fun) * dolfin.dx
    res = dolfin.Function(V)
    solver = dolfin.LocalSolver(a,L)
    solver.solve_local_rhs(res)
    return res


def setup_bullseye_sim(bullseye_mesh, fun_arr):
    V = FunctionSpace(bullseye_mesh, "DG", 0)
    dm = V.dofmap()
    sfun = MeshFunction("size_t", bullseye_mesh, 2, bullseye_mesh.domains())

    funcs = []
    for time in range(len(fun_arr)):

        fun_tmp = Function(V)
        arr = fun_arr[time]

        for region in range(17):

                vertices = []

                for cell in cells(bullseye_mesh):

                    if sfun.array()[cell.index()] == region+1:

                        verts = dm.cell_dofs(cell.index())

                        for v in verts:
                            # Find the correct vertex index 
                            if v not in vertices:
                                vertices.append(v)

                fun_tmp.vector()[vertices] = arr[region]
        funcs.append(Vector(fun_tmp.vector()))
    return funcs
    
