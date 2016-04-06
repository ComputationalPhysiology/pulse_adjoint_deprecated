#!/bin/bash

#SBATCH --job-name=AdjointContractionArray
#
# Project:
#SBATCH --account=NN9249K
#
# Wall clock limit:
#SBATCH --time=01:00:00
#


#Send emails for start, stop, fail, etc...
#SBATCH --output=slurmfiles/save_patient_data-%j.out


## Set up job environment:
source /cluster/bin/jobsetup

set -o errexit # exit on errors

module purge   # clear any inherited modules
module load gcc/5.1.0
module load openmpi.gnu/1.8.8
module load cmake/3.1.0
export CC=gcc
export CXX=g++
export FC=gfortran
export F77=gfortran
export F90=gfortran

ulimit -S -s unlimited

arrayrun $1-$2 save_patient_data.slurm