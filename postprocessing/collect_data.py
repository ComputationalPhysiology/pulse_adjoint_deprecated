"""
Copy all the result data into one single file

"""
import h5py, os
import numpy as np


def collect_data(main_dir, res_dir, alphas, reg_pars):
    
    all_results_fname = "/".join([main_dir, "all_results.h5"])
    file_format = "a" if os.path.isfile(all_results_fname) else "w"
    all_results = h5py.File(all_results_fname , file_format)

    for a in alphas:

        # Create a group if it does not allready exist
        if not "alpha_{}".format(a) in all_results.keys():
            all_results.create_group("alpha_{}".format(a))

        for l in reg_pars:
            
            res_file = "/".join([res_dir.format(a,l), "result.h5"])
            if not os.path.isfile(res_file):
                continue

            print res_file
            result = h5py.File(res_file, "r")
                

            # If it allready exist
            if "reg_par_{}".format(l) in all_results["alpha_{}".format(a)].keys():
                # Delete the old one
                del all_results["alpha_{}/reg_par_{}".format(a, l)]

            # Copy over the new one
            h5py.h5o.copy(result.id, "alpha_{}".format(a), 
                          all_results.id, "alpha_{}/reg_par_{}".format(a, l))

            # Save crashes
            crash_file = "/".join([res_dir.format(a,l), "gamma_crash.h5"])
            if os.path.isfile(crash_file):
                # If it allreade exists
                if "crash" in all_results["alpha_{}/reg_par_{}".format(a, l)].keys():
                    # Delete it
                    del all_results["alpha_{}/reg_par_{}/crash".format(a, l)]

                # Open the crash file
                crash = h5py.File(crash_file, "r")
                all_results.create_group("alpha_{}/reg_par_{}/crash".format(a, l))

                # For each crash point
                for c in crash.keys():
                    # Copy 
                    h5py.h5o.copy(crash.id, c, all_results.id, 
                                  "alpha_{}/reg_par_{}/crash/{}".format(a, l, c))
                    

            result.close()
    all_results.close()
    print "Saved to {}".format(all_results_fname)


def collect_real_data():

    patient = "Impact_p16_i43"
    

    sim_file_str_main_dir = "results/patient_{}".format(patient)
    sim_file_dir = "/".join([sim_file_str_main_dir, 
                             "/alpha_{}/regpar_{}/med_res"])

    alphas = [i/100.0 for i in range(10)] + [i/10.0 for i in range(1,11)]
    reg_pars = [0.0]
    collect_data(sim_file_str_main_dir, sim_file_dir, alphas, reg_pars)
    
def collect_synthetic_data():
    
    patient = "Impact_p16_i43"
    noise = True

    synth_file_str_main_dir = "results/synthetic_noise_{}/patient_{}".format(noise, patient)
    synth_file_str = "/".join([synth_file_str_main_dir, 
                         "/alpha_{}/regpar_{}/med_res/result.h5"])


    alphas = [i/100.0 for i in range(10)] + [i/10.0 for i in range(1,11)]
    reg_pars = [0.0]
    collect_data(synth_file_str_main_dir, synth_file_str, alphas, reg_pars)

    alphas = [0.03, 0.4]
    reg_pars = np.logspace(-10,-1, 10).tolist() + np.multiply(5, np.logspace(-10, -1, 10)).tolist()
    collect_data(synth_file_str_main_dir, synth_file_str, alphas, reg_pars)
    



if __name__ == "__main__":
    # collect_synthetic_data()
    collect_real_data()
