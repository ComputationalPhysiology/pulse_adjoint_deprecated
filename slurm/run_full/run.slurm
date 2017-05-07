#!/bin/bash

#SBATCH --job-name=SubName
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
#SBATCH --ntasks-per-node=1

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
OUTDIR=$(python outfile.py $INPUT "outdir")
OUTPUT=$OUTDIR"/result.h5"
echo $OUTDIR
#GAMMACRASH=$OUTDIR"/gamma_crash.h5"
MESH=$(python outfile.py $INPUT "mesh")
PRESSURE=$(python outfile.py $INPUT "pressure")
#ECHO=$(python outfile.py $INPUT "echo")
#MESHBASE=${MESH##*/}
## Copy input files to the work directory:
cp run.py $SCRATCH 
cp $INPUT $SCRATCH
cp $MESH $SCRATCH
cp $PRESSURE $SCRATCH
#cp $ECHO $SCRATCH
## Make sure the results are copied back to the submit directory (see Work Directory below):
#chkfile $OUTPUT
#chkfile $GAMMACRASH
mkdir -p $OUTDIR

cleanup "cp $SCRATCH/result.h5 $OUTDIR/result.h5"
cleanup "cp $SCRATCH/input.yml $OUTDIR/input.yml"
cleanup "cp $SCRATCH/output.log $OUTDIR/output.log"
## Do some work:
cd $SCRATCH



mpirun python run.py $INPUT $OUTPUT
