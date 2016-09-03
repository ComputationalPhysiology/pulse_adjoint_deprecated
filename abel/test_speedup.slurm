#!/bin/bash
NCORES=$1
#SBATCH --job-name=PulseAdjointSpeedUp"$NCORES"
#
# Project:
#SBATCH --account=NN9249K
#
# Wall clock limit:
#SBATCH --time=96:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node="$NCORES"

#Send emails for start, stop, fail, etc...
#SBATCH --mail-type=END
#SBATCH --output=slurmfiles/impact-%j.out
#SBATCH --mail-user=henriknf@simula.no

## Set up job environment:
# source /cluster/bin/jobsetup

# set -o errexit # exit on errors

# module purge   # clear any inherited modules
# module load gcc/5.1.0
# module load openmpi.gnu/1.8.8
# module load cmake/3.1.0
# export CC=gcc
# export CXX=g++
# export FC=gfortran
# export F77=gfortran
# export F90=gfortran

# ulimit -S -s unlimited

SUBMITDIR="."

OUTDIR=$2
ID=$3

INPUT=$SUBMITDIR"/input/file_"$ID".yml"
OUTPUT=$OUTDIR"/result.h5"
GAMMACRASH=$OUTDIR"/gamma_crash.h5"

# cp run.py $SCRATCH 
# cp $INPUT $SCRATCH
# ## Make sure the results are copied back to the submit directory (see Work Directory below):
# chkfile $OUTPUT
# chkfile $GAMMACRASH
# ## Do some work:
# cd $SCRATCH


if [ $NCORES -eq 1 ]; then
    python run.py $INPUT $OUTPUT
else
    mpirun python run.py $INPUT $OUTPUT
fi

