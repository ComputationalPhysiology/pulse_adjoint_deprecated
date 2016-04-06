#!/bin/bash

#SBATCH --job-name=SavePatientDataRun-%j
#
# Project:
#SBATCH --account=NN9249K
#
# Wall clock limit:
#SBATCH --time=00:15:00


#Send emails for start, stop, fail, etc...
#SBATCH --output=slurmfiles/save_patient_datarun-%j.out


## Set up job environment:
source /cluster/bin/jobsetup
set -o errexit # exit on errors


ulimit -S -s unlimited
module purge   # clear any inherited modules
module load gcc/5.1.0
module load openmpi.gnu/1.8.8
module load cmake/3.1.0
export CC=gcc
export CXX=g++
export FC=gfortran
export F77=gfortran
export F90=gfortran



# Input file
INPUT=$SUBMITDIR"/input/file_"$TASK_ID".yml"

# Output file
OUTDIR=$(python outfile.py $INPUT)
OUTPUT=$OUTDIR"/result.h5"

## Copy input files to the work directory:
cp run.py $SCRATCH 
cp $INPUT $SCRATCH
## Make sure the results are copied back to the submit directory (see Work Directory below):
chkfile $OUTPUT
chkfile $GAMMACRASH
## Do some work:
cd $SCRATCH


python save_patient_data.py $INPUT $OUTPUT
