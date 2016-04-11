from postprocess_utils import *
import pickle
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



def plot_L_curves_alpha(data, outdir):

    cm = plt.cm.get_cmap('RdYlBu')

    # Plot I_strain vs I_vol for different alphas
    # and reg_par = 0.0
    I_strain = []
    I_vol = []
    alphas = []
    reg_par = 0.0
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
    alpha = 0.3
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


def plot_volume(data, outdir_str):
    
    
    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            path = "/".join([outdir, "volume.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            simulated_volume = merge_passive_active(data, alpha, reg_par, "volume")
            n = len(simulated_volume)
            synthetic_volume = data["measured"]["volume"][:n]
            
            x = np.linspace(0,100,n)
            plot_curves(x, 
                        [synthetic_volume, simulated_volume], 
                        ["Measured","Simulated"], 
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

            
            simulated_strains, n = merge_passive_active(data, alpha, reg_par, 
                                                        "strain", return_n = True)
            
            
            measured_strains = get_strain_partly(data["measured"]["strain"], n)
            
            
            labels = ["Measured", "Simulated"]
            # The order you put it in here, is the order it should be labeled

            
            s_min, s_max = get_min_max_strain(measured_strains, simulated_strains)
            strains = strain_to_arrs(measured_strains, simulated_strains)
            

            plot_canvas(strains, s_min, s_max, path, labels)



def simulation(data, kwargs, outdir_str):

    mesh = kwargs["mesh"]
    strain_markers = kwargs["strain_markers"]
    time_stamps = kwargs["time_stamps"] 

    sm = Function(kwargs["marker_space"], name = "strain_markers")
    sm.vector()[:] = strain_markers.array()


    for alpha in data["active"].keys():
        for reg_par in data["active"][alpha].keys():

            # Create directory
            outdir = outdir_str.format(alpha, reg_par)
            # path = "/".join([outdir, "volume.pdf"])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            simulated_gammas = merge_passive_active(data, alpha, reg_par, "gammas")
            simulated_displacements = merge_passive_active(data, alpha, reg_par, "displacements")

   
            # stress = Function(kwargs["stress_space"], 
            #                   name="stress")
            # work = Function(kwargs["stress_space"], 
            #                 name="cardiac_work")
            gamma = Function(kwargs["gamma_space"], 
                             name="gamma")
            u = Function(kwargs["displacement_space"], 
                         name="displacement")

    
            fname = "simulation_{}.vtu"
            path = outdir + "/" + fname        
            for i,t in enumerate(time_stamps):

                # stress.vector()[:] = stresses[i]
                gamma.vector()[:] = simulated_gammas[i]
                u.vector()[:] = simulated_displacements[i]
                # work.vector()[:] = works[i]

                add_stuff(mesh, path.format(i), gamma, sm, u)

            write_pvd(outdir+"/simulation.pvd", fname, time_stamps)

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

            
            simulated_gamma = merge_passive_active(data, alpha, reg_par, "gammas")
            simulated_states = merge_passive_active(data, alpha, reg_par, "states")

           
            # Simulated results
            newmesh_sim = Mesh(mesh)
            new_spaces_sim = init_spaces(newmesh_sim)
            sm_sim = Function(new_spaces_sim["marker_space"], name = "strain_markers")
            sm_sim.vector()[:] = strain_markers.array()
            # stress_sim = Function(new_spaces_sim["stress_space"], name="simulated_stress")
            gamma_sim = Function(new_spaces_sim["gamma_space"], name="simulated_gamma")
            u_prev, u_current, state, d, fa = setup_moving_mesh(kwargs["state_space"], newmesh_sim)


            fname = "simulation_moving_sim_{}.vtu"
            path = outdir + "/" + fname
            for i,t in enumerate(time_stamps):

                state.vector()[:] = simulated_states[i]
                u,p = state.split()
                fa.assign(u_current, u)
                d.vector()[:] = u_current.vector()[:] - u_prev.vector()[:]
                newmesh_sim.move(d)

                # stress_sim.vector()[:] = stresses_sim[i]
                gamma_sim.vector()[:] = simulated_gamma[i]
                sm_sim.vector()[:] = strain_markers.array()

                add_stuff(newmesh_sim, path.format(i), gamma_sim, sm_sim)

                u_prev.assign(u_current)

            write_pvd(outdir+"/simulation_moving_sim.pvd", fname, time_stamps)


            


def save_moving_simulation(sim_data, orig_spaces, outdir, params):

    mesh = orig_spaces["mesh"]
    strain_markers = orig_spaces["strain_markers"]
    time_stamps = orig_spaces["time_stamps"] 

    # Functions on the original mesh
    stresses = sim_data["stresses"] 
    gammas = sim_data["gammas"] 
    states = sim_data["states"] 
    works = sim_data["cardiac_work"]
    displacements = sim_data["displacements"] 
    
    # Create a new mesh
    newmesh = Mesh(mesh)
    spaces = init_spaces(newmesh, 
                         params["gamma_space"])

    # Create new functions
    sm = Function(spaces["marker_space"], 
                  name = "strain_markers")
    sm.vector()[:] = strain_markers.array()
    stress = Function(spaces["stress_space"], 
                      name="stress")
    work = Function(spaces["stress_space"], 
                    name="cardiac_work")
    gamma = Function(spaces["gamma_space"], 
                     name="gamma")


    u_prev, u_current, state, d, fa = setup_moving_mesh(orig_spaces["state_space"], newmesh)

    fname = "simulation_moving_{}.vtu"
    path = outdir + "/" + fname
    for i,t in enumerate(time_stamps):

        state.vector()[:] = states[i]
        u,p = state.split()
        fa.assign(u_current, u)

        d.vector()[:] = u_current.vector()[:] - \
          u_prev.vector()[:]
        newmesh.move(d)

        stress.vector()[:] = stresses[i]
        gamma.vector()[:] = gammas[i]
        work.vector()[:] = works[i]
        sm.vector()[:] = strain_markers.array()

        add_stuff(newmesh, path.format(i), stress, 
                  gamma, sm, work)

        u_prev.assign(u_current)

    write_pvd(outdir+"/simulation_moving.pvd", 
              fname, time_stamps)





def postprocess(data, kwargs, params):

    print Text.blue("\nStart postprocessing")
    outdir_main = "/".join([params["outdir"], "alpha_{}", "regpar_{}"])

    # print Text.purple("Plot volume")
    # outdir = "/".join([outdir_main, "volume"])
    # plot_volume(data.copy(), outdir)

    # print Text.purple("Plot strain")
    # outdir = "/".join([outdir_main, "strain"])
    # plot_strain(data.copy(), outdir)

    print Text.purple("Save simulation")
    outdir = "/".join([outdir_main, "simulation"])
    simulation(data.copy(), kwargs, outdir)


    print Text.purple("Save moving simulation")
    outdir = "/".join([outdir_main, "simulation_moving"])
    simulation_moving(data.copy(), kwargs, outdir)

    print Text.purple("Plot misfit L-curves")
    plot_L_curves_alpha(data.copy(), params["outdir"])
    plot_L_curves_lambda(data.copy(), params["outdir"])
            


def initialize(params, alpha_regpars):

    
    # Load patient class
    patient = initialize_patient_data(params["Patient_parameters"], False)
    

    # Get simulated data and some extra stuff
    data, kwargs = get_all_data(params, patient, alpha_regpars)
    


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

    
    # Path to results
    params["sim_file"] = "results/patient_{}/results.h5".format(params["Patient_parameters"]["patient"])
    params["outdir"] = os.path.dirname(params["sim_file"])

    from itertools import product
    alphas = [i/10.0 for i in range(11)] + [i/100.0 for i in range(11)] 
    # alphas = [0.3]

    # reg_pars = [0.0]
    # reg_pars = np.logspace(-4,-2, 11, dtype = float).tolist() + [0.0]
    reg_pars = np.logspace(-10,-1, 10).tolist() + \
      np.multiply(5, np.logspace(-10, -1, 10)).tolist() + \
      np.logspace(-4,-2, 11).tolist() + [0.0]
   

    alpha_regpars = list(product(np.unique(alphas), 
                                 np.unique(reg_pars)))

    
    simulated_data, kwargs = initialize(params, alpha_regpars)
    postprocess(simulated_data, kwargs, params)
    


if __name__ == "__main__":
     main()
   
    
