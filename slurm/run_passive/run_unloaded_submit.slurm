#!/bin/bash

#SBATCH --job-name=UnloadedImpact
#
# Project:
#SBATCH --account=NN9249K
#
# Wall clock limit:
#SBATCH --time=96:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=4G

#Send emails for start, stop, fail, etc...
#SBATCH --mail-type=END
#SBATCH --output=slurmfiles/pah-%j.out
#SBATCH --mail-user=henriknf@simula.no

mkdir -p slurmfiles
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

arrayrun $1-$2 run_unloaded.slurm
