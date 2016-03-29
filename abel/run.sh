#SBATCH --job-name=AdjointContractionRun$(TASK_ID)
#
# Project:
#SBATCH --account=NN9249K
#
# Wall clock limit:
#SBATCH --time=00:01:05
#
# Max memory usage:
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#Send emails for start, stop, fail, etc...
#SBATCH --mail-type=END

#SBATCH --mail-user=henriknf@simula.no

## Set up job environment:
# source /cluster/bin/jobsetup
# module purge   # clear any inherited modules
# set -o errexit # exit on errors


# module load gcc/4.9.2
# module load openmpi.gnu/1.8.4
# module load cmake/2.8.9

# ulimit -S -s unlimited
export TASK_ID=1
export SUBMITDIR="."
# Input file
INPUT=$SUBMITDIR"/input/file_"$TASK_ID".yml"
# Output directory
OUTPUT=$(python outfile.py $INPUT)
echo $OUTPUT
# Create output directory if not exists

# Output file


## Copy input files to the work directory:
# cp run.py $SCRATCH 
# cp $INPUT $SCRATCH
## Make sure the results are copied back to the submit directory (see Work Directory below):
# chkfile $OUTPUT
## Do some work:
# cd $SCRATCH
# mpirun -n 4 
python run.py $INPUT $OUTPUT

