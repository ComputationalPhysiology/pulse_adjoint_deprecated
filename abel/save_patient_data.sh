#!/bin/bash

## Set up job environment:
# source /cluster/bin/jobsetup
# set -o errexit # exit on errors


# ulimit -S -s unlimited
# module purge   # clear any inherited modules
# module load gcc/5.1.0
# module load openmpi.gnu/1.8.8
# module load cmake/3.1.0
# export CC=gcc
# export CXX=g++
# export FC=gfortran
# export F77=gfortran
# export F90=gfortran



export TASK_ID=1
export SUBMITDIR="."
# Input file
INPUT=$SUBMITDIR"/input/file_"$TASK_ID".yml"

# Output file
OUTDIR=$(python outfile.py $INPUT)
OUTPUT=$OUTDIR"/result.h5"


python save_patient_data.py $INPUT $OUTPUT
