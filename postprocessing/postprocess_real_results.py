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
from pulse_adjoint.setup_optimization import RegionalGamma
figdir = "/home/finsberg/src/adjoint_contraction/article/figures/reproducible_plots/data"
save_data_to_figdir = False

try:
    from mesh_generation.create_segmeted_sim import save_segmented_surfaces
    save_seg_surfaces = True
except:
    save_seg_surfaces = False


def save_pickle_to_fig_dir(res_dir, alphas, reg_pars):

    data = {}
    for alpha in alphas:
        data[alpha] = {}
        for reg_par in reg_pars:
            res_path = res_dir + "/results_{}_{}.p".format(alpha, reg_par)
            with open( res_path, "rb" ) as f:
                d = pickle.load(f)

            data[alpha][reg_par] = d
    
    try:
        with open( figdir+"/misfit_simulation.p", "rb" ) as output:
            f = pickle.load(output)

        update_dict(f,data)

    except:
        f = data

    
    with open( figdir+"/misfit_simulation.p", "wb" ) as output:
        pickle.dump(f, output, pickle.HIGHEST_PROTOCOL)


######### CREATE IMAGES ##############
def plot_L_curves_alpha(data, outdir):

    cm = plt.cm.get_cmap('RdYlBu')

    # Plot I_strain vs I_vol for different alphas
    # and reg_par = 0.0
    I_strain = []
    I_vol = []
    alphas = []
    reg_par = 1.0
    for alpha in data["active"].keys():
        
        misfit = data["active"][alpha][reg_par]["misfit"]
        I_strain.append(np.mean(misfit["I_strain_optimal"]))
        I_vol.append(np.mean(misfit["I_volume_optimal"]))
        alphas.append(float(alpha))
        
    fig = plt.figure()
    ax = fig.gca() 

    ax.set_yscale('log')
    ax.set_xscale('log')

    s = ax.scatter(I_vol, I_strain, c = alphas, cmap = cm, s = 40)

    ax.set_ylabel(r"$\overline{I}_{\mathrm{strain}}$")
    ax.set_xlabel(r"$\overline{I}_{\mathrm{vol}}$")

    cbar = plt.colorbar(s)
    cbar.set_label(r"$\alpha$")
    fig.savefig(outdir + "/l_curve_alpha.pdf")
    

def plot_L_curves_lambda(data, outdir):
    
    # Plot I_strain+I_vol vs grad gamma with fixed
    # alphas and different reg_pars
    I_misfit = []
    reg_pars = []
    gamma_gradient = []
    alpha = 0.9
    for reg_par in data["active"][alpha].keys():

        if reg_par != 0.0:
            misfit = data["active"][alpha][reg_par]["misfit"]
            I_misfit.append(np.mean(misfit["I_strain_optimal"]) + 
                            np.mean(misfit["I_volume_optimal"]))
            gamma_gradient.append(np.mean(data["active"][alpha][reg_par]["gamma_gradient"]))

            reg_pars.append(float(reg_par))

    fig = plt.figure()
    ax = fig.gca() 

    ax.set_yscale('log')
    ax.set_xscale('log')

    cm = plt.cm.get_cmap('RdYlBu')


    s = ax.scatter(gamma_gradient, I_misfit, c = reg_pars, 
                   cmap = cm, s = 40, norm=mpl.colors.LogNorm())
  

    ax.set_ylabel(r"$\overline{I}_{\mathrm{strain}} + \overline{I}_{\mathrm{vol}}$")
    ax.set_xlabel(r"$\overline{\| \nabla \gamma \|}^2$")
    
    cbar = plt.colorbar(s)
    cbar.set_label(r"$\lambda$")

    fig.savefig(outdir + "/l_curve_lambda.pdf")

def plot_misfit(data, outdir_str):


    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            path = "/".join([outdir, "misfit.pdf"])

            misfit = data["active"][alpha][reg_par]["misfit"]

            num_contract_points = data["active"][alpha][reg_par]["num_points"]
            num_passsive_points = data["passive"]["num_points"]

            if 0:#save_data_to_figdir:
                with open( figdir+"/simulated_misfit_real.yml", "wb" ) as output:
                # with open( figdir+"/simulated_misfit.yml", "wb" ) as output:
                    f = {"I_strain_initial": misfit["I_strain_initial"], 
                         "I_strain_optimal": misfit["I_strain_optimal"], 
                         "I_volume_initial": misfit["I_volume_initial"], 
                         "I_volume_optimal": misfit["I_volume_optimal"], 
                         "num_contract_points": num_contract_points, 
                         "num_passsive_points": num_passsive_points}
                    yaml.dump(f, output, default_flow_style=False)
                # exit()
            start = 100*(1-float(num_contract_points)/(num_contract_points + num_passsive_points))
            x = np.linspace(start,100, len(misfit["I_strain_initial"]))
            
            fig, ax1 = plt.subplots()
            ax1.semilogy(x, misfit["I_strain_initial"], 'b--')
            ax1.semilogy(x, misfit["I_strain_optimal"], 'b-')
            ax1.set_xlabel('$\%$ cardiac cycle')
            # Make the y-axis label and tick labels match the line color.
            ax1.set_ylabel('$I_{\mathrm{strain}}$', color='b')
            for tl in ax1.get_yticklabels():
                tl.set_color('b')


            ax2 = ax1.twinx()
            ax2.semilogy(x, misfit["I_volume_initial"], 'r--')
            ax2.semilogy(x, misfit["I_volume_optimal"], 'r-')
            ax2.set_ylabel('$I_{\mathrm{volume}}$', color='r')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')

            fig.savefig(path, bbox_inches='tight')

def print_passive_misfit(data):
    """
    Print the initial and optimal misfit 
    functional from the passive optimization
    """
    
    print data["passive"]["misfit"]
    if save_data_to_figdir:
        with open( figdir+"/simulated_passive_misfit_real.yml", "wb" ) as output:
            yaml.dump(data["passive"]["misfit"], output, default_flow_style=False)


def plot_volume(data, outdir_str):
    """
    Plot simulated versus measured volumes
    """
    
    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            path = "/".join([outdir, "volume.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            simulated_volume = merge_passive_active(data, alpha, reg_par, "volume")
            n = len(simulated_volume)
            measured_volume = data["measured"]["volume"][:n]
            
            x = np.linspace(0,100,n)
            fig = plt.figure()
            ax = fig.gca() 

            ax.plot(x, simulated_volume, marker = "o", label = "Simualted")
            ax.plot(x, measured_volume, marker = "o", label = "Measured")

            ax.legend(loc = 'upper left')
            ax.set_ylabel(r"Volume (ml)")
            ax.set_xlabel(r"$\%$ cardiac cylce")
            

            fig.savefig(outdir.format(alpha, reg_par)+
                        "/volume.pdf")

def plot_fiber_stress(data, kwargs, outdir):
    """
    Plot regional fiber stress
    """
    
    basal_im = plt.imread("bullseye/bullseye_base_act.png")
    mid_im = plt.imread("bullseye/bullseye_mid_act.png")
    apical_im = plt.imread("bullseye/bullseye_apical_act.png")

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            ys = merge_passive_active(data, alpha, reg_par, "fiber_stresses")

            y = Function(kwargs["stress_space"])
            y_reg = get_regional(kwargs["dx"], y, ys)


            path = outdir.format(alpha, reg_par) \
              + "/regional_fiber_stress.pdf" 
            plot_regional(y_reg, path, r"Fiber stress (kPa)")


def plot_fiber_work(data, kwargs, outdir):
    """
    Plot regional work done in the direction
    of the fibers.
    """

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            ys = merge_passive_active(data, alpha, reg_par, "fiber_work")

            y = Function(kwargs["stress_space"])
            y_reg = get_regional(kwargs["dx"], y, ys)


            path = outdir.format(alpha, reg_par) \
              + "/regional_fiber_work.pdf" 
            plot_regional(y_reg, path, r"Fiber Work")

def plot_work(data, kwargs, outdir):
    """
    Plot total regional work.
    """

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            ys = merge_passive_active(data, alpha, reg_par, "work")

            y = Function(kwargs["stress_space"])
            y_reg = get_regional(kwargs["dx"], y, ys)


            path = outdir.format(alpha, reg_par) \
              + "/regional_work.pdf" 
            plot_regional(y_reg, path, r"Work")

def plot_fiber_strain(data, kwargs, outdir):
    """
    Plot regional fiber strain
    """

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            ys = merge_passive_active(data, alpha, reg_par, "fiber_strains")

            y = Function(kwargs["stress_space"])
            y_reg = get_regional(kwargs["dx"], y,ys)


            path = outdir.format(alpha, reg_par) \
              + "/regional_fiber_strain.pdf" 
            plot_regional(y_reg, path, r"Fiber Strain")

def plot_gamma(data, kwargs, outdir):
    """
    Plot regional gamma.
    """

    basal_im = plt.imread("bullseye/bullseye_base_act.png")
    mid_im = plt.imread("bullseye/bullseye_mid_act.png")
    apical_im = plt.imread("bullseye/bullseye_apical_act.png")

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            ys = merge_passive_active(data, alpha, reg_par, "gammas")

            y = Function(kwargs["gamma_space"])
            y_reg = get_regional(kwargs["dx"], y, ys)


            path = outdir.format(alpha, reg_par) \
              + "/regional_gamma.pdf" 
            plot_regional(y_reg, path, r"Gamma")
            
    

def plot_pv_loop(data, outdir_str):
    """
    Plot measured and simulated PV loops.
    """
    
    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            path = "/".join([outdir, "volume.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            simulated_volume = merge_passive_active(data, alpha, reg_par, "volume")
            n = len(simulated_volume)
            measured_volume = data["measured"]["volume"][:n]

            # Convert pressure to kPa
            pressure = data["measured"]["pressure"][:n]
            
  
            fig = plt.figure()
            ax = fig.gca() 

            ax.plot(simulated_volume, pressure, marker = "o", label = "Simualted")
            ax.plot(measured_volume, pressure,  marker = "o", label = "Measured")
            
            ax.legend(loc = 'upper left')
            ax.set_ylabel(r"Pressure (kPA)")
            ax.set_xlabel(r"Volume (ml)")

            if save_data_to_figdir:
                with open( figdir+"/simulated_pv_loop.yml", "wb" ) as output:
                    f = {"measured_volume": measured_volume, 
                         "simulated_volume":simulated_volume, 
                         "pressure":pressure}
                    yaml.dump(f, output, default_flow_style=False)


            fig.savefig(outdir.format(alpha, reg_par)+
                        "/pv_loop.pdf")

            


def plot_strain(data, kwargs, outdir_str):
    """
    Plot the simulated and measured strains.
    """

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            path = "/".join([outdir, "strains.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)


            # Plot the strains how they are computed in the simulation
            simulated_strains, n = merge_passive_active(data, alpha, reg_par, 
                                                        "strain", return_n = True)
            
            
            measured_strains = get_strain_partly(data["measured"]["strain"], n)
            
            if save_data_to_figdir:
                with open( figdir+"/simulated_strain.yml", "wb" ) as output:
                    f = {"simulated_strains": simulated_strains, 
                         "measured_strains":measured_strains}
                    yaml.dump(f, output, default_flow_style=False)

            labels = ["Measured", "Simulated"]
            # The order you put it in here, is the order it should be labeled
            plot_strains2(simulated_strains, measured_strains, outdir)
            
            s_min, s_max = get_min_max_strain(measured_strains, simulated_strains)
            strains = strain_to_arrs(measured_strains, simulated_strains)
            plot_canvas(strains, s_min, s_max, path, labels)

            
            # Recompute the strains according to the original reference in echopac
            outdir_ = "/".join([outdir, "strains_orig_ref"])
            if not os.path.exists(outdir_):
                os.makedirs(outdir_)
            ref = kwargs["num_points"] - kwargs["passive_filling_begins"]
            strains = data["measured"]["strain"]
            simulated_strains = recompute_strains_to_original_reference(simulated_strains, ref)
            measured_strains = recompute_strains_to_original_reference(measured_strains, ref)
            plot_strains2(simulated_strains, measured_strains, outdir_)
            
            
            

            
            

def plot_vph(data, outdir_str):

    sns.set_palette("husl")
    sns.set_style("whitegrid")
    # sns.set_style("ticks")
    sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})
    # mpl.rcParams.update({'figure.autolayout': True})
    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 60} 

    # mpl.rc('font', **font)
    plt.rc('text', usetex=True)
    plt.rc('font', family = 'serif')
    rcParams['text.usetex']=True
    rcParams['text.latex.unicode']=True

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par) + "/vph"
            # path = "/".join([outdir, "strains.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)


            simulated_strains, n = merge_passive_active(data, alpha, reg_par, 
                                                        "strain", return_n = True)
            
            measured_strains = get_strain_partly(data["measured"]["strain"], n)

            simulated_volume = merge_passive_active(data, alpha, reg_par, "volume")
            n = len(simulated_volume)
            measured_volume = data["measured"]["volume"][:n]
            pressure = data["measured"]["pressure"][:n]
            
            # pv_loop_data = data["simulated_pv_loop"]

            sim_strains = [simulated_strains["longitudinal"][4],
                               simulated_strains["longitudinal"][2],
                               simulated_strains["longitudinal"][3],
                               simulated_strains["longitudinal"][6]]

            meas_strains = [measured_strains["longitudinal"][4],
                                measured_strains["longitudinal"][2],
                                measured_strains["longitudinal"][3],
                                measured_strains["longitudinal"][6]]

            x = np.linspace(0,100, len(sim_strains[0]))

            with sns.axes_style("darkgrid"):

                # Plot PV loop
                fig = plt.figure()
                ax = fig.gca() 

                # pressure = np.add(pressure, 2.8)
                # Plot simulated loop
                l1 = ax.plot(simulated_volume, 
                                 pressure, sns.xkcd_rgb["pale red"], lw=3,
                                 label = "Simulated")

                # Plot measured loop
                l2 = ax.plot(measured_volume, 
                                 pressure, sns.xkcd_rgb["denim blue"], lw=3,
                                label = "Measured")


                lines = [l1[0], l2[0]]
                labels = ["Measured", "Simulated"]
                lgd = fig.legend( lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0.5,1.04))
                
                # ax.legend(loc = 'upper left')
                ax.set_ylabel(r"Pressure (kPA)")
                ax.set_xlabel(r"Volume (ml)")


                fig.savefig(outdir+"/pv_loop.pdf", bbox_extra_artists=(lgd,))#bbox_inches='tight')

                plt.close()

    
                # Plot strains
                fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')

                axes[0,0].plot(x, sim_strains[0], sns.xkcd_rgb["pale red"], lw=3)
                axes[0,0].plot(x, meas_strains[0], sns.xkcd_rgb["denim blue"], lw=3)
                axes[0,0].set_title("Basal Inferior")

                axes[1,0].plot(x, sim_strains[1], sns.xkcd_rgb["pale red"], lw=3)
                axes[1,0].plot(x, meas_strains[1], sns.xkcd_rgb["denim blue"], lw=3)
                axes[1,0].set_title("Basal Anteroseptal")

                axes[0,1].plot(x, sim_strains[2], sns.xkcd_rgb["pale red"], lw=3)
                axes[0,1].plot(x, meas_strains[2], sns.xkcd_rgb["denim blue"], lw=3)
                axes[0,1].set_title("Basal Septum")
        
                l1 = axes[1,1].plot(x, sim_strains[3], sns.xkcd_rgb["pale red"], lw=3)
                l2 = axes[1,1].plot(x, meas_strains[3], sns.xkcd_rgb["denim blue"], lw=3)
                axes[1,1].set_title("Basal Lateral")

                # Labels
                fig.text(0.5, 0.01, r'$\%$ cardiac cycle', ha='center')
                fig.text(0.01, 0.5, r'Longitudinal Strain', va='center', rotation='vertical')



                # Legend
                lines = [l1[0], l2[0]]
                labels = ["Measured", "Simulated"]
                lgd = fig.legend( lines, labels, loc = 'upper center', ncol=2, bbox_to_anchor=(0.5,1.04))

                fig.tight_layout()
                fig.savefig(outdir+"/strain.pdf", bbox_extra_artists=(lgd,))
                plt.close()

######## CREATE PARAVIEW FILES ###########
def simulation(data, kwargs, outdir_str):
    """
    Create a pvd file with a simualtion to 
    be viewed in paraview.
    
    The simlation contains a visualization of
    gamma, fiber_stress, fiber_strain, work, 
    fiber_work, p (hydrostatic pressure), I1 and I4f

    Displacements are stored in a separate xdmf file.
    """

    mesh = kwargs["mesh"]
    strain_markers = kwargs["strain_markers"]
    time_stamps = kwargs["time_stamps"] 

    sm = Function(kwargs["marker_space"], name = "strain_markers")
    sm.vector()[:] = strain_markers.array()


    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            
            gammas = merge_passive_active(data, alpha, reg_par, "gammas")
            fiber_stresses = merge_passive_active(data, alpha, reg_par, "fiber_stresses")
            fiber_strains = merge_passive_active(data, alpha, reg_par, "fiber_strains")
            works = merge_passive_active(data, alpha, reg_par, "work")
            fiber_works = merge_passive_active(data, alpha, reg_par, "fiber_work")
            ps = merge_passive_active(data, alpha, reg_par, "p")
            I1s = merge_passive_active(data, alpha, reg_par, "I1")
            I4fs = merge_passive_active(data, alpha, reg_par, "I4f")
            Effs = merge_passive_active(data, alpha, reg_par, "Eff") 
            Tffs = merge_passive_active(data, alpha, reg_par, "Tff") 

            n = len(gammas)
  
               
            fiber_stress = Function(kwargs["stress_space"], 
                                    name="fiber_stress")
            fiber_strain = Function(kwargs["stress_space"], 
                                    name="fiber_strain")
            work = Function(kwargs["stress_space"], 
                            name="work")
            fiber_work = Function(kwargs["stress_space"], 
                            name="fiber_work")

            
            if kwargs["gamma_space"].dim() == 17:
                gamma_space = FunctionSpace(mesh, "DG", 0)
                rg = RegionalGamma(strain_markers)
            else:
                gamma_space = kwargs["gamma_space"]
            
            gamma = Function(gamma_space, 
                             name="gamma")
            p = Function(kwargs["pressure_space"], 
                         name="hydrostatic_pressure")
            I1 = Function(kwargs["pressure_space"], 
                         name="I1")
            I4f = Function(kwargs["pressure_space"], 
                         name="I4_f")
            Eff = Function(kwargs["quad_space"], name = "Eff")
            Tff = Function(kwargs["quad_space"], name = "Tff")

            fname = "simulation_{}.vtu"
            path = outdir + "/" + fname        
            for i,t in enumerate(time_stamps[:n]):

                if kwargs["gamma_space"].dim() == 17:
                    rg.assign(gammas[i])
                    gamma.vector()[:] = rg.project("DG_0").vector()
                else:
                    gamma.vector()[:] = gammas[i]


                fiber_stress.vector()[:] = fiber_stresses[i]
                fiber_strain.vector()[:] = fiber_strains[i]
                
                work.vector()[:] = works[i]
                fiber_work.vector()[:] = fiber_works[i]
                p.vector()[:] = ps[i]
                I1.vector()[:] = I1s[i]
                I4f.vector()[:] = I4fs[i]

                Eff.vector()[:] = Effs[i]
                Tff.vector()[:] = Tffs[i]

                add_stuff(mesh, path.format(i), 
                          gamma, sm, p, I1, I4f, 
                          fiber_stress, fiber_strain, 
                          work, fiber_work, Eff, Tff)

            write_pvd(outdir+"/simulation.pvd", fname, time_stamps)


            # Save displacements
            simulated_displacements = merge_passive_active(data, alpha, reg_par, "displacements")
            u = Function(kwargs["displacement_space"], name="displacement")
            path = outdir + "/" + "displacement.xdmf"
            disp_file = XDMFFile(mpi_comm_world(), path)
            for i,t in enumerate(time_stamps[:n]):
                u.vector()[:] = simulated_displacements[i]
                disp_file << u, float(t)
            
            del disp_file

def simulation_moving(data, kwargs, outdir_str):
    """
    Create a pvd file with a moving simualtion to 
    be viewed in paraview.
    
    The simlation contains a visualization of
    gamma, fiber_stress, fiber_strain, work, 
    fiber_work, p (hydrostatic pressure), I1 and I4f
    """

    strain_markers = kwargs["strain_markers"]
    time_stamps = kwargs["time_stamps"] 
            
    
    mesh = kwargs["mesh"]
    

    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            if save_data_to_figdir:
                with open( outdir+"/time_stamps.yml", "wb" ) as output:
                    f = {"time_staps": time_stamps}
                    yaml.dump(f, output, default_flow_style=False)

            states = merge_passive_active(data, alpha, reg_par, "states")
            gammas = merge_passive_active(data, alpha, reg_par, "gammas")
            fiber_stresses = merge_passive_active(data, alpha, reg_par, "fiber_stresses")
            fiber_strains = merge_passive_active(data, alpha, reg_par, "fiber_strains")
            works = merge_passive_active(data, alpha, reg_par, "work")
            fiber_works = merge_passive_active(data, alpha, reg_par, "fiber_work")
            ps = merge_passive_active(data, alpha, reg_par, "p")
            I1s = merge_passive_active(data, alpha, reg_par, "I1")
            I4fs = merge_passive_active(data, alpha, reg_par, "I4f")
            Effs = merge_passive_active(data, alpha, reg_par, "Eff") 
            Tffs = merge_passive_active(data, alpha, reg_par, "Tff")

            n = len(gammas)

            newmesh = Mesh(mesh)
            new_spaces = init_spaces(newmesh)
            sm = Function(new_spaces["marker_space"], name = "strain_markers")
            sm.vector()[:] = strain_markers.array()
            
            fiber_stress = Function(new_spaces["stress_space"], 
                                    name="fiber_stress")
            fiber_strain = Function(new_spaces["stress_space"], 
                                    name="fiber_strain")
            work = Function(new_spaces["stress_space"], 
                            name="work")
            fiber_work = Function(new_spaces["stress_space"], 
                            name="fiber_work")

            if kwargs["gamma_space"].dim() == 17:
                gamma_space = FunctionSpace(mesh, "DG", 0)
                rg = RegionalGamma(strain_markers)
            else:
                gamma_space = kwargs["gamma_space"]

            gamma = Function(gamma_space, name="gamma")
            p = Function(new_spaces["pressure_space"], 
                         name="hydrostatic_pressure")
            I1 = Function(new_spaces["pressure_space"], 
                          name="I1")
            I4f = Function(new_spaces["pressure_space"], 
                           name="I4_f")
            Eff = Function(new_spaces["quad_space"], name = "Eff")
            Tff = Function(new_spaces["quad_space"], name = "Tff")


            u_prev, u_current, state, d, fa = setup_moving_mesh(kwargs["state_space"], newmesh)


            fname = "simulation_moving_sim_{}.vtu"
            path = outdir + "/" + fname
            for i,t in enumerate(time_stamps[:n]):

                state.vector()[:] = states[i]
                u,_= state.split()
                fa.assign(u_current, u)
                d.vector()[:] = u_current.vector()[:] - u_prev.vector()[:]
                newmesh.move(d)

                if kwargs["gamma_space"].dim() == 17:
                    rg.assign(gammas[i])
                    gamma.vector()[:] = rg.project("DG_0").vector()
                else:
                    gamma.vector()[:] = gammas[i]

                fiber_stress.vector()[:] = fiber_stresses[i]
                fiber_strain.vector()[:] = fiber_strains[i]
                
                work.vector()[:] = works[i]
                fiber_work.vector()[:] = fiber_works[i]
                p.vector()[:] = ps[i]
                I1.vector()[:] = I1s[i]
                I4f.vector()[:] = I4fs[i]

                Eff.vector()[:] = Effs[i]
                Tff.vector()[:] = Tffs[i]
                
                add_stuff(newmesh, path.format(i), 
                          gamma, sm, p, I1, I4f, 
                          fiber_stress, fiber_strain, 
                          work, fiber_work, Eff, Tff)


                u_prev.assign(u_current)

            write_pvd(outdir+"/simulation_moving_sim.pvd", fname, time_stamps)

def simulation_from_segementation(data, params, outdir_str):
    """
    Create a simulation with snap shots of the
    segmented surfaces from Echo Pac
    """
    try:
        # from mesh_generation.create_ply_files import HeartSurfaces
        from mesh_generation.mesh_utils import ECHODIR, get_time_stamps
        from patient_data.scripts.data import PHASES
        from mesh_generation.generate_mesh import MeshConstructor, setup_mesh_parameters
    except:
        print "Unable to create simualtion of segementation"
        return

    

    name = params["Patient_parameters"]["patient"]
    echo_path = os.path.join(ECHODIR, "US_sim{}.h5".format(name))
    start = PHASES[name]["passive_filling_begins"]

    def mesh_params(time):
        mparams = setup_mesh_parameters(name, "med_res")
        mparams["time"] = time
        mparams["ply_dir"] = os.path.abspath(params["outdir"] + "/ply")
        if not os.path.exists(mparams["ply_dir"]):
            os.makedirs(mparams["ply_dir"])
        return mparams
        
        
        

    time_stamps = get_time_stamps(echo_path)
    time_stamps = np.subtract(time_stamps, time_stamps[0])
    n = len(time_stamps)

    # for i in range(24, n):

    #     mparams = mesh_params(i)
    #     # from IPython import embed; embed()
    #     # exit()
    #     M = MeshConstructor(mparams)
    #     M.generate_mesh()


    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            vtp_folder = outdir + "/vtp_files"

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            if not os.path.exists(vtp_folder):
                os.makedirs(vtp_folder)

            for i in range(n):


                # params = mesh_params(i, outdir)
                # M = MeshConstructor(params)
                # M.generate_mesh()
                
                # Create a mesh for this time step
                # H = HeartSurfaces(name, i, "lv", outdir)
                # H.create_lv_ply_files()
                mparams = mesh_params(i)

                # from IPython import embed; embed()
                # exit()

                # LV endo
                # polydata_endo = ply_to_polydata(H.endo_lv_outfile)
                polydata_endo = ply_to_polydata(mparams["ply_dir"]+"/{}_lv/endo_lv_{}.ply".format(name, i))
                fname_endo = vtp_folder + "/endo_lv_{}".format((start-i) % n) 
                write_to_vtp(fname_endo, polydata_endo)

                # LV epi
                # polydata_epi = ply_to_polydata(H.epi_lv_outfile)
                polydata_epi = ply_to_polydata(mparams["ply_dir"]+"/{}_lv/epi_lv_{}.ply".format(name, i))
                fname_epi = vtp_folder + "/epi_lv_{}".format((start-i) % n) 
                write_to_vtp(fname_epi, polydata_epi)
                

            write_pvd(outdir+"/endo.pvd", "vtp_files/endo_lv_{}.vtp", time_stamps)
            write_pvd(outdir+"/epi.pvd", "vtp_files/epi_lv_{}.vtp", time_stamps)

        
            
########## MAIN FUNCTIONS #############
def postprocess(data, kwargs, params, outdir_main):
    """
    Main file for postprocessing
    """
    

    print Text.blue("\nStart postprocessing")
    

    # plot_vph(data.copy(), outdir_main)
    # exit()
    # print Text.purple("\nSave segmented surfaces")
    # outdir = "/".join([outdir_main, "segmentation"])
    # simulation_from_segementation(data, params, outdir)
    # exit()
    
    print Text.purple("Plot volume")
    plot_volume(data.copy(), outdir_main)
 
    print Text.purple("Plot gamma")
    plot_gamma(data.copy(), kwargs, outdir_main)

    print Text.purple("Plot fiber stress")
    plot_fiber_stress(data.copy(), kwargs, outdir_main)

    print Text.purple("Plot fiber strain")
    plot_fiber_strain(data.copy(), kwargs, outdir_main)

    print Text.purple("Plot Work")
    plot_work(data.copy(), kwargs, outdir_main)

    print Text.purple("Plot Fiber Work")
    plot_fiber_work(data.copy(), kwargs, outdir_main)

    print Text.purple("Plot misfit")
    plot_misfit(data.copy(), outdir_main)
    print_passive_misfit(data.copy())
    
    

    print Text.purple("Plot PV loop")
    plot_pv_loop(data.copy(), outdir_main)

    print Text.purple("Plot strain")
    plot_strain(data.copy(), kwargs, outdir_main)
    # exit()

    print Text.purple("Save simulation")
    outdir = "/".join([outdir_main, "simulation"])
    simulation(data.copy(), kwargs, outdir)


    print Text.purple("Save moving simulation")
    outdir = "/".join([outdir_main, "simulation_moving"])
    simulation_moving(data.copy(), kwargs, outdir)

    # print Text.purple("Plot misfit L-curves")
    # plot_L_curves_alpha(data.copy(), params["outdir"])
    # plot_L_curves_lambda(data.copy(), params["outdir"])


def postprocess_single(params):
    # Load patient class
    patient = initialize_patient_data(params["Patient_parameters"], False)

    load_geometry_and_microstructure_from_results(patient,params)

    data, kwargs = load_single_result(params, patient)

    return data, kwargs


def initialize(params, alpha_regpars):
    """
    Collect the results.
    """

    # Load patient class
    patient = initialize_patient_data(params["Patient_parameters"], False)
   

    # Get simulated data and some extra stuff
    data, kwargs = get_all_data(params, patient, alpha_regpars)
    
    if save_data_to_figdir:
        pass
        # with open( figdir+"/simulated_data.yml", "wb" ) as output:
        #     volume = data[]
        #     f = {"lambdas": reg_pars, "I_misfit":I_misfit, "gamma_gradient":gamma_gradient}
        #     yaml.dump(f, output, default_flow_style=False)


    return data, kwargs




def main():
    set_log_active(False)

    params = setup_adjoint_contraction_parameters()
    
    # Turn of annotation
    parameters["adjoint"]["stop_annotating"] = True


    patients = ["Impact_p12_i45",
                  "Impact_p10_i45",
                  "Impact_p8_i56",
                  "Impact_p9_i49",
                  "Joakim",
                  "Sjur_10",
                  "Sam_12",
                  "CRID-pas_ESC",
                  "Impact_p19_i55"]
        
    patients = ["Impact_p16_i43"]

    for patient in patients:
        # params["Patient_parameters"]["patient"] = "Impact_p15_i38"
        params["Patient_parameters"]["patient"] = patient
        # params["Patient_parameters"]["patient"] = "Sam_12"
        # params["Patient_parameters"]["patient"] = "Sjur_10"
        params["Patient_parameters"]["resolution"] = "low_res"
    
        # params["gamma_space"] = "regional"
        params["gamma_space"] = "CG_1"
        # params["gamma_space"] = "R_0"
        params["alpha_matparams"] = 1.0

        # wild = "_active_stress"
        wild = "_segbase_old"
        
        if params["gamma_space"] == "CG_1":
            params["sim_file"] = "results/patient_{}{}{}/results.h5".format(params["Patient_parameters"]["patient"],
                                                                                params["Patient_parameters"]["resolution"], wild)
        elif params["gamma_space"] == "R_0":
            params["sim_file"] = "results/patient_{}_scalar_{}/results.h5".format(params["Patient_parameters"]["patient"],
                                                                                      params["Patient_parameters"]["resolution"])

        elif params["gamma_space"] == "regional":
            params["sim_file"] = "results/patient_{}_regional_{}_l/results.h5".format(params["Patient_parameters"]["patient"],
                                                                                          params["Patient_parameters"]["resolution"])
        else:
            raise IOError("No results for gamma space = {}".format(params["gamma_space"]))




        params["alpha"] = 0.9
        params["reg_par"] = 0.01
       
       
        params["Patient_parameters"]["fiber_angle_epi"] = 20
        params["Patient_parameters"]["fiber_angle_endo"] = 20
        
        params["sim_file"] = "results/patient_{}/alpha_{}/regpar_{}/fendo{}_fepi{}/result.h5".format(params["Patient_parameters"]["patient"], params["alpha"], params["reg_par"], params["Patient_parameters"]["fiber_angle_endo"], params["Patient_parameters"]["fiber_angle_epi"])
        print params["sim_file"]
        params["outdir"] = os.path.dirname(params["sim_file"])

        
        # alphas = [0.9]
        # alphas = [0.9, 0.95, 0.99, 0.999]
        # reg_pars = [0.1]
        # reg_pars = [1.0, 0.1, 0.01, 10.0]
        
        # from itertools import product
        # alpha_regpars = list(product(np.unique(alphas), 
                                         # np.unique(reg_pars)))

        
        simulated_data, kwargs = postprocess_single(params)
        outdir_main = params["outdir"]

        
        # simulated_data, kwargs = initialize(params, alpha_regpars)
        # outdir_main = "/".join([params["outdir"], "alpha_{}", "regpar_{}"])
        
        postprocess(simulated_data, kwargs, params, outdir_main)
    


if __name__ == "__main__":
     main()
   
    
