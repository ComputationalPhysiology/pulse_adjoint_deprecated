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


#ulimit -S -s unlimited

# Input file
INPUT=$SUBMITDIR"/input/file_"$TASK_ID".p"

# Output file
OUTPUT=$(python outfile.py $INPUT)
#touch $OUTPUT

## Copy input files to the work directory:
cp run.py $SCRATCH 
cp $INPUT $SCRATCH
## Make sure the results are copied back to the submit directory (see Work Directory below):
chkfile $OUTPUT
## Do some work:
cd $SCRATCH


mpirun python run.py $INPUT $OUTPUT
