#!/bin/bash


#SBATCH --job-name=AdjointContractionRun"$TASK_ID"
#
# Project:
#SBATCH --account=NN9249K
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

#Send emails for start, stop, fail, etc...
#SBATCH --mail-type=END
#SBATCH --output=slurmfiles/impact-%j.out
#SBATCH --mail-user=henriknf@simula.no


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
GAMMACRASH=$OUTDIR"/gamma_crash.h5"

## Copy input files to the work directory:
cp run.py $SCRATCH 
cp $INPUT $SCRATCH
## Make sure the results are copied back to the submit directory (see Work Directory below):
chkfile $OUTPUT
chkfile $GAMMACRASH
## Do some work:
cd $SCRATCH


mpirun python run.py $INPUT $OUTPUT
