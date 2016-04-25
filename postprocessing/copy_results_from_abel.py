#!/usr/bin/env python
" A script for copying files from abel to local machine"
import os, h5py, subprocess, tempfile, shutil, sys
import numpy as np


HOST = "henriknf@abel.uio.no"
MAIN_PATH = "{}:~/mypackages/lib/python2.7/site-packages/campass/abel/results".format(HOST)

def copy_h5py_file(tmpfile, local_path):

    if os.path.isfile(local_path):

        old_file = h5py.File(local_path, "a")
        new_file = h5py.File(tmpfile, "r")


        for alpha in new_file.keys():
            
            if not alpha in old_file.keys():
                h5py.h5o.copy(new_file.id, alpha, 
                              old_file.id, alpha)

            else:
                for reg_par in new_file[alpha].keys():
                    if not reg_par in old_file[alpha].keys():
                        h5py.h5o.copy(new_file.id, "/".join([alpha, reg_par]), 
                                      old_file.id, "/".join([alpha, reg_par]))

                    else:
                        if not "active_contraction" in new_file[alpha][reg_par].keys():
                            continue

                        if not "active_contraction" in old_file[alpha][reg_par].keys():
                            h5py.h5o.copy(new_file.id, "/".join([alpha, reg_par, "active_contraction"]), 
                                              old_file.id, "/".join([alpha, reg_par, "active_contraction"]))
 
                        for point in new_file[alpha][reg_par]["active_contraction"].keys():
                            if not point in old_file[alpha][reg_par]["active_contraction"].keys():
                                h5py.h5o.copy(new_file.id, "/".join([alpha, reg_par, "active_contraction", point]), 
                                              old_file.id, "/".join([alpha, reg_par, "active_contraction", point]))


        old_file.close()
        new_file.close()
    else:
        shutil.move(tmpfile, local_path)

def copy_synth():
    
    patients = ["Impact_p16_i43"]
    noise = True

    for patient in patients:
        # Path to the results on abel
        abel_path = "{}/synthetic_noise_{}/patient_{}/all_results.h5".format(MAIN_PATH, noise, patient)
        

        print abel_path
        # create a temporary directory
        tmpdir = tempfile.mkdtemp()
        
        # Copy the data to a temporary file
        tmpfile = os.path.join(tmpdir, 'results.h5')
        print tmpfile
        
        subprocess.call(["scp", abel_path, tmpfile])

        # Path to where the results show be copied to
        local_path = "results/synthetic_noise_{}/patient_{}/results.h5".format(noise, patient)
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Copy everything from the tmpfile to the local file
        # and make sure that we do not delete any old data
        copy_h5py_file(tmpfile, local_path)


def copy_real(scalar = False):
    
    patients = ["Impact_p16_i43"]

    scalar_str = "_scalar" if scalar else ""        

    for patient in patients:
        # Path to the results on abel
        abel_path = "{}/real{}/patient_{}/all_results.h5".format(MAIN_PATH, scalar_str, patient)
        print abel_path
        
        # create a temporary directory
        tmpdir = tempfile.mkdtemp()

        # Copy the data to a temporary file
        tmpfile = os.path.join(tmpdir, 'results.h5')
        print tmpfile
        subprocess.call(["scp", abel_path, tmpfile])

        # Path to where the results show be copied to
        local_path = "results/patient_{}{}/results.h5".format(patient, scalar_str)
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Copy everything from the tmpfile to the local file
        # and make sure that we do not delete any old data
        copy_h5py_file(tmpfile, local_path)
        print "Saved to ", local_path

def copy_real_scalar():
    pass

def test1():
    tmpfile = "results/patient_Impact_p16_i43/all_results.h5"
    local_path = "results/patient_Impact_p16_i43/all_results2.h5"
    copy_h5py_file(tmpfile, local_path)

def test2():

    abel_path = "results/patient_Impact_p16_i43/all_results.h5"
    print abel_path
    
    
    tmpdir = tempfile.mkdtemp()

    # Copy the data to a temporary file
    tmpfile = os.path.join(tmpdir, 'results.h5')
    subprocess.call(["scp", abel_path, tmpfile])
    local_path = "results/patient/results.h5"
    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Copy everything from the tmpfile to the local file
    # and make sure that we do not delete any old data
    copy_h5py_file(tmpfile, local_path)
    print local_path
    
        

if __name__ == "__main__":
    
    print sys.argv
    
    if len(sys.argv) > 1:
        data = sys.argv[1]
        if len(sys.argv) == 3:
            scalar = sys.argv[2]
        else:
            scalar = False
        

        if data == "synth":
            copy_synth()

        elif data == "real":                
            copy_real(scalar)


