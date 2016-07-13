#!/usr/bin/env python
# Copyright (C) 2016 Henrik Finsberg
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
from postprocess_utils import *
import pickle

figdir = "/home/finsberg/src/adjoint_contraction/article/figures/reproducible_plots/data"

def print_gamma_err(data, kwargs, outdir):
    
    gamma_space = kwargs["gamma_space"]
    mesh = kwargs["mesh"]
    dx = kwargs["dx"]

    with open(outdir + "/gamma_error.txt", "wb") as f:

        f.write("\n"+"#"*5+" Gamma Error "+"#"*5+"\n")

        for alpha in data["active"].keys():

            f.write("\nAlpha = {}\n".format(alpha))

            
            f.write("lambda\t\tL2 err\t\tRegional Err\tInf Err\n")
            for reg_par in np.sort(data["active"][alpha].keys()):

                simulated_gammas =  merge_passive_active(data, alpha, reg_par, "gammas")
                synthetic_gammas = data["synthetic"]["gammas"]

                gamma_err_l2 = get_errornorm_lst(synthetic_gammas, 
                                                 simulated_gammas, 
                                                 gamma_space, mesh, "L2")

                

                gamma_err_reg = get_regional_norm_lst(synthetic_gammas, 
                                                      simulated_gammas, 
                                                      gamma_space, dx)

                gamma_err_inf = get_maxnorm_lst(synthetic_gammas,
                                                simulated_gammas, 
                                                gamma_space)

                f.write("{:.2e}\t{:.2e}\t{:.2e}\t{:.2e}\n".format(reg_par, 
                                                                np.max(gamma_err_l2), 
                                                                np.max(gamma_err_reg), 
                                                                np.max(gamma_err_inf)))

def print_displacement_err(data, kwargs, outdir):
    
    displacement_space = kwargs["displacement_space"]
    mesh = kwargs["mesh"]
    dx = kwargs["dx"]

    with open(outdir + "/displacement_error.txt", "wb") as f:

        f.write("\n"+"#"*5+" Displacement Error "+"#"*5+"\n")

        for alpha in data["active"].keys():

            f.write("\nAlpha = {}\n".format(alpha))

            
            f.write("lambda\t\tL2 err\t\tInf Err\n")
            for reg_par in np.sort(data["active"][alpha].keys()):

                simulated_displacements =  merge_passive_active(data, alpha, reg_par, "displacements")
                synthetic_displacements = data["synthetic"]["displacements"]

                disp_err_l2 = get_errornorm_lst(synthetic_displacements, 
                                                 simulated_displacements, 
                                                 displacement_space, mesh, "L2")

                

                disp_err_inf = get_maxnorm_lst(synthetic_displacements,
                                               simulated_displacements, 
                                               displacement_space)

                f.write("{:.2e}\t{:.2e}\t{:.2e}\n".format(reg_par, 
                                                          np.max(disp_err_l2),  
                                                          np.max(disp_err_inf)))



def simulation(data, kwargs, outdir_str):

    
    strain_markers = kwargs["strain_markers"]
    time_stamps = kwargs["time_stamps"] 
    mesh = kwargs["mesh"]

    sm = Function(kwargs["marker_space"], name = "strain_markers")
    sm.vector()[:] = strain_markers.array()

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            # path = "/".join([outdir, "volume.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            
            simulated_gamma = merge_passive_active(data, alpha, reg_par, "gammas")
            simulated_stresses = merge_passive_active(data, alpha, reg_par, "stresses")
            simulated_work = merge_passive_active(data, alpha, reg_par, "work")
            n = len(simulated_gamma)

            synthetic_gammas = data["synthetic"]["gammas"]
            synthetic_stresses = data["synthetic"]["stresses"]
            synthetic_work = data["synthetic"]["work"]


            gamma_sim = Function(kwargs["gamma_space"], name="simulated_gamma")
            stress_sim = Function(kwargs["stress_space"], name="simulated_stress")
            work_sim = Function(kwargs["stress_space"], name="simulated_work")
            
            gamma_synth = Function(kwargs["gamma_space"], name="synthetic_gamma")
            stress_synth = Function(kwargs["stress_space"], name="synthetic_stress")
            work_synth = Function(kwargs["stress_space"], name="synthetic_work")
            
            
            gamma_diff = Function(kwargs["gamma_space"], name="difference_gamma")
            stress_diff = Function(kwargs["stress_space"], name="difference_stress")
            work_diff = Function(kwargs["stress_space"], name="difference_work")

            fname = "simulation_{}.vtu"
            path = outdir + "/" + fname        
            for i,t in enumerate(time_stamps[:n]):

                work_sim.vector()[:] = simulated_work[i]
                stress_sim.vector()[:] = simulated_stresses[i]
                gamma_sim.vector()[:] = simulated_gamma[i]
               

                work_synth.vector()[:] = synthetic_work[i]
                stress_synth.vector()[:] = synthetic_stresses[i]
                gamma_synth.vector()[:] = synthetic_gammas[i]
                

                
                gamma_diff.assign(project(gamma_sim-gamma_synth))
                stress_diff.assign(project(stress_sim-stress_synth))
                work_diff.assign(project(work_sim-work_synth))


                add_stuff(mesh, path.format(i), sm,  \
                          gamma_sim, gamma_synth, gamma_diff, \
                          work_sim, work_synth, work_diff, 
                          stress_sim, stress_synth, stress_diff)

            write_pvd(outdir+"/simulation.pvd", fname, time_stamps)

            # Save displacement
            simulated_displacements = merge_passive_active(data, alpha, reg_par, "displacements")
            synthetic_displacements = data["synthetic"]["displacements"]
            u_sim = Function(kwargs["displacement_space"], name="simulated displacement")
            u_synth = Function(kwargs["displacement_space"], name="synthetic displacement")
            u_diff = Function(kwargs["displacement_space"], name="difference displacement")
            
            disp_file_sim = XDMFFile(mpi_comm_world(), outdir + "/" + "displacement_sim.xdmf")
            disp_file_synth = XDMFFile(mpi_comm_world(), outdir + "/" + "displacement_synth.xdmf")
            disp_file_diff = XDMFFile(mpi_comm_world(), outdir + "/" + "displacement_diff.xdmf")
            for i,t in enumerate(time_stamps[:n]):

                u_sim.vector()[:] = simulated_displacements[i]
                disp_file_sim << u_sim, float(t)

                u_synth.vector()[:] = synthetic_displacements[i]
                disp_file_synth << u_synth, float(t)

                u_diff.assign(project(u_sim-u_synth))
                disp_file_diff << u_diff, float(t)

            
            del disp_file_sim
            del disp_file_synth
            del disp_file_diff
            


def simulation_moving(data, kwargs, outdir_str):

    strain_markers = kwargs["strain_markers"]
    time_stamps = kwargs["time_stamps"] 
    mesh = kwargs["mesh"]
    

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            # path = "/".join([outdir, "volume.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            
            # Simulated results
            simulated_gamma = merge_passive_active(data, alpha, reg_par, "gammas")
            simulated_states = merge_passive_active(data, alpha, reg_par, "states")
            simulated_stresses = merge_passive_active(data, alpha, reg_par, "stresses")
            simulated_work = merge_passive_active(data, alpha, reg_par, "work")
            n = len(simulated_gamma)
            
            newmesh_sim = Mesh(mesh)
            new_spaces_sim = init_spaces(newmesh_sim)
            sm_sim = Function(new_spaces_sim["marker_space"], name = "strain_markers")
            sm_sim.vector()[:] = strain_markers.array()

            work_sim = Function(new_spaces_sim["stress_space"], name="simulated_work")
            stress_sim = Function(new_spaces_sim["stress_space"], name="simulated_stress")
            gamma_sim = Function(new_spaces_sim["gamma_space"], name="simulated_gamma")
            u_prev, u_current, state, d, fa = setup_moving_mesh(kwargs["state_space"], newmesh_sim)


            fname = "simulation_moving_sim_{}.vtu"
            path = outdir + "/" + fname
            for i,t in enumerate(time_stamps[:n]):

                state.vector()[:] = simulated_states[i]
                u,p = state.split()
                fa.assign(u_current, u)
                d.vector()[:] = u_current.vector()[:] - u_prev.vector()[:]
                newmesh_sim.move(d)

                work_sim.vector()[:] = simulated_work[i]
                stress_sim.vector()[:] = simulated_stresses[i]
                gamma_sim.vector()[:] = simulated_gamma[i]
                sm_sim.vector()[:] = strain_markers.array()

                add_stuff(newmesh_sim, path.format(i), gamma_sim, 
                          sm_sim, work_sim, stress_sim)

                u_prev.assign(u_current)

            write_pvd(outdir+"/simulation_moving_sim.pvd", fname, time_stamps)




            # Synthetic results
            synthetic_gammas = data["synthetic"]["gammas"]
            synthetic_states= data["synthetic"]["states"]
            synthetic_stresses = data["synthetic"]["stresses"]
            synthetic_work = data["synthetic"]["work"]


            newmesh_synth = Mesh(mesh)
            new_spaces_synth = init_spaces(newmesh_synth)
            sm_synth = Function(new_spaces_synth["marker_space"], name = "strain_markers")
            sm_synth.vector()[:] = strain_markers.array()

            work_synth = Function(new_spaces_synth["stress_space"], name="synthetic_work")
            stress_synth = Function(new_spaces_synth["stress_space"], name="synthetic_stress")
            gamma_synth = Function(new_spaces_synth["gamma_space"], name="synthetic_gamma")
            u_prev, u_current, state, d, fa = setup_moving_mesh(kwargs["state_space"], newmesh_synth)

            fname = "simulation_moving_synth_{}.vtu"
            path = outdir + "/" + fname
            for i,t in enumerate(time_stamps):

                state.vector()[:] = synthetic_states[i]
                u,p = state.split()
                fa.assign(u_current, u)
                d.vector()[:] = u_current.vector()[:] - u_prev.vector()[:]
                newmesh_synth.move(d)

                work_synth.vector()[:] = synthetic_work[i]
                stress_synth.vector()[:] = synthetic_stresses[i]
                gamma_synth.vector()[:] = synthetic_gammas[i]

                sm_synth.vector()[:] = strain_markers.array()
                domain = newmesh_synth.ufl_domain()

                add_stuff(newmesh_synth, path.format(i), gamma_synth, sm_synth)

                u_prev.assign(u_current)

            write_pvd(outdir+"/simulation_moving_synth.pvd", fname, time_stamps)
    
def plot_L_curves_alpha(data, outdir):

    cm = plt.cm.get_cmap('RdYlBu')

    # Plot I_strain vs I_vol for different alphas
    # and reg_par = 0.0
    I_strain = []
    I_vol = []
    alphas = []
    reg_par = 0.0
    for alpha in data["active"].keys():
        
        if reg_par not in data["active"][alpha].keys():
            continue

        misfit = data["active"][alpha][reg_par]["misfit"]
        I_strain.append(np.mean(misfit["I_strain_optimal"])/51)
        I_vol.append(np.mean(np.sqrt(misfit["I_volume_optimal"])))
        alphas.append(float(alpha))
        
    
    with open( figdir+"/lcurve_alpha_synth.yml", "wb" ) as output:
        f = {"alphas": alphas, "I_strain":I_strain, "I_vol":I_vol}
        yaml.dump(f, output, default_flow_style=False)

    fig = plt.figure()
    ax = fig.gca() 

    ax.set_yscale('log')
    ax.set_xscale('log')

    s = ax.scatter(I_vol, I_strain, c = alphas, cmap = cm, s = 40)

    for i, txt in enumerate(alphas):
        if txt in [0.0, 0.1, 0.4, 1.0]:
            ax.annotate(str(txt), (I_vol[i],I_strain[i]), size = 14)
        

    ax.set_ylabel(r"$\overline{I}_{\mathrm{strain}}$")
    ax.set_xlabel(r"$\overline{I}_{\mathrm{vol}}$")

    cbar = plt.colorbar(s)
    cbar.set_label(r"$\alpha$")
    fig.savefig(outdir + "/l_curve_alpha.pdf")
    plt.close()

def plot_L_curves_lambda(data, outdir):

    cm = plt.cm.get_cmap('RdYlBu')

    # Plot I_strain+I_vol vs grad gamma with fixed
    # alphas and different reg_pars
    I_misfit = []
    reg_pars = []
    gamma_gradient = []
    alpha = 0.8
    if alpha not in data["active"].keys():
        return

    for reg_par in data["active"][alpha].keys():
        misfit = data["active"][alpha][reg_par]["misfit"]
        I_misfit.append(np.mean(misfit["I_strain_optimal"][:-1])/51 + 
                        np.mean(np.sqrt(misfit["I_volume_optimal"][:-1])))
        gamma_gradient.append(np.mean(data["active"][alpha][reg_par]["gamma_gradient"][:-1]))

        reg_pars.append(float(reg_par))


    with open( figdir+"/lcurve_lambda_synth.yml", "wb" ) as output:
        f = {"lambdas": reg_pars, "I_misfit":I_misfit, "gamma_gradient":gamma_gradient}
        yaml.dump(f, output, default_flow_style=False)

    fig = plt.figure()
    ax = fig.gca() 

    ax.set_yscale('log')
    ax.set_xscale('log')

    s = ax.scatter(gamma_gradient, I_misfit, c = reg_pars, 
                   cmap = cm, s = 40, norm=mpl.colors.LogNorm())
    
    # for i, txt in enumerate(reg_pars):
    #     if i%2:
    #         textstr = '$%.1e$'%(txt)
            # ax.text(gamma_gradient[i], I_misfit[i], textstr, size = 14)

    ax.set_ylabel(r"$\overline{I}_{\mathrm{strain}} + \overline{I}_{\mathrm{vol}}$")
    ax.set_xlabel(r"$\overline{\| \nabla \gamma \|}^2$")
    
    cbar = plt.colorbar(s)
    cbar.set_label(r"$\lambda$")
   

    fig.savefig(outdir + "/l_curve_lambda.pdf")
    plt.close()    


def plot_volume(data, outdir_str):
    
    
    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            path = "/".join([outdir, "volume.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            simulated_volume, n = merge_passive_active(data, alpha, reg_par, "volume", True)
            synthetic_volume = data["synthetic"]["volume"][:n]
            
            x = np.linspace(0,100,len(synthetic_volume))
            plot_curves(x, 
                        [synthetic_volume, simulated_volume], 
                        ["Synthetic","Simulated"], 
                        "",#"Volume",
                        "$\%$ cardiac cycle",
                        "Volume(ml)",
                        path)
            
def plot_strain(data, outdir_str):
    

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            path = "/".join([outdir, "strains.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            simulated_strains, n = merge_passive_active(data, alpha, reg_par, "strain", True)
            synthetic_strains = get_strain_partly(data["synthetic"]["strain"], n)
            
            
            labels = ["Synthetic", "Simulated"]
            # The order you put it in here, is the order it should be labeled

            
            s_min, s_max = get_min_max_strain(synthetic_strains, simulated_strains)
            strains = strain_to_arrs(synthetic_strains, simulated_strains)

            plot_canvas(strains, s_min, s_max, path, labels)
    

def postprocess(data, kwargs, params):

    print Text.blue("\nStart postprocessing")
    outdir_main = "/".join([params["outdir"], "alpha_{}", "regpar_{}"])


    # print Text.purple("Plot volume")
    # outdir = "/".join([outdir_main, "volume"])
    # plot_volume(data, outdir)

    # print Text.purple("Plot strain")
    # outdir = "/".join([outdir_main, "strain"])
    # plot_strain(data, outdir)
        
 
    # print Text.purple("Save simulation")
    # outdir = "/".join([outdir_main, "simulation"])
    # simulation(data.copy(), kwargs, outdir)

    # print Text.purple("Save moving simulation")
    # outdir = "/".join([outdir_main, "simulation_moving"])
    # simulation_moving(data.copy(), kwargs, outdir)

    print Text.purple("Plot misfit L-curves")
    plot_L_curves_alpha(data.copy(), params["outdir"])
    plot_L_curves_lambda(data.copy(), params["outdir"])
    
    # print Text.purple("Save gamma error")
    # print_gamma_err(data.copy(), kwargs, params["outdir"])
    
    # print Text.purple("Save displacement error")
    # print_displacement_err(data.copy(), kwargs, params["outdir"])
    
    
                
def initialize(params, alpha_regpars):
    
    # Load patient class
    patient = initialize_patient_data(params["Patient_parameters"], True)
    
    # Get simulated data and some extra stuff
    data, kwargs = get_all_data(params, patient, alpha_regpars, synthetic_data=True)

    return data, kwargs


def main():
    set_log_active(False)

    params = setup_adjoint_contraction_parameters()
    
    # Turn of annotation
    parameters["adjoint"]["stop_annotating"] = True


    params["Patient_parameters"]["patient"] = "Impact_p16_i43"
    # params["Patient_parameters"]["patient"] = "CRID-pas_ESC"
    params["Patient_parameters"]["resolution"] = "med_res"
    
    params["gamma_space"] = "CG_1"
    params["alpha_matparams"] = 1.0
    params["noise"] = True
    
    # Path to results
    params["sim_file"] = "results/new_fun_synthetic_noise_{}/patient_{}/results.h5".format(params["noise"], 
                                                                                   params["Patient_parameters"]["patient"])

    params["outdir"] = os.path.dirname(params["sim_file"])

    from itertools import product
    alphas = [i/10.0 for i in range(11)] #+ [i/100.0 for i in range(11)] 
    # alphas = [0.4]

    # reg_pars = [0.0]
    reg_pars = np.logspace(-4,1, 6).tolist() + [0.0] #+ np.linspace(0,30,11)[1:].tolist()
    # reg_pars = np.logspace(-10,-1, 10).tolist() + \
    #   np.multiply(5, np.logspace(-10, -1, 10)).tolist() + \
    #   np.logspace(-4,-2, 11).tolist() + [0.0] + \
    #   np.linspace(0.005, 0.02, 10).tolist() + \
    #   np.linspace(0,100,11).tolist()[1:] + [1.0, 5.0]

    # reg_pars = np.logspace(-4,-1, 4).tolist() + \
          # np.linspace(0.005, 0.02, 10).tolist() + [0.0]
   
      
    alpha_regpars = list(product(np.unique(alphas), 
                                 np.unique(reg_pars)))

    
    data, kwargs = initialize(params, alpha_regpars)
    
    postprocess(data, kwargs, params)
    


if __name__ == "__main__":

    main()
    # if len(sys.argv) < 2:
    #     # noise = "drift"
    #     noise = None
    #     main(noise = noise)

    # else:
    #     alpha = sys.argv[1]
    #     reg_par = sys.argv[2]
    #     main(alpha, reg_par)
    
