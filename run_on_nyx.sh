#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=experiment

# Define, how many nodes you need. Here, we ask for 1 node.
# Each node has 16 or 20 CPU cores.
#SBATCH --nodes=1
##SBATCH --exclusive
#SBATCH --ntasks=10
# You can further define the number of tasks with --ntasks-per-*
# See "man sbatch" for details. e.g. --ntasks=4 will ask for 4 cpus.

# Define, how long the job will run in real time. This is a hard cap meaning
# that if the job runs longer than what is written here, it will be
# force-stopped by the server. If you make the expected time too long, it will
# take longer for the job to start. Here, we say the job will take 5 minutes.
#              d-hh:mm:ss
#SBATCH --time=0-24:00:00

# Define the partition on which the job shall run. May be omitted.
#SBATCH --partition compute

# How much memory you need.
# --mem will define memory per node and
# --mem-per-cpu will define memory per CPU/core. Choose one of those.
##SBATCH --mem-per-cpu=1500MB
##SBATCH --mem=5GB    # this one is not in effect, due to the double hash

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julianstastny@gmail.com
# You may not place any commands before the last SBATCH directive

module add singularity
export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
# singularity shell /home/jstastny/numpyro_latest.sif


# Define and create a unique scratch directory for this job
SCRATCH_DIRECTORY=/ptmp/${USER}/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was submitted from
cp ${SLURM_SUBMIT_DIR}/*.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/*.pkl ${SCRATCH_DIRECTORY}

# This is where the actual work is done. In this case, the script only waits.
# The time command is optional, but it may give you a hint on how long the
# command worked
# bash
# conda activate base
# eval "$(conda shell.bash hook)"

# singularity exec /home/jstastny/numpyro_latest.sif echo "which python" | bash && cd ${SCRATCH_DIRECTORY} && python3 run_experiments.py
# singularity shell /home/jstastny/numpyro_latest.sif -c "bash && cd ${SCRATCH_DIRECTORY} && python3 run_experiments.py &> stdout && /bin/bash -norc"
singularity exec /home/jstastny/numpyro_latest.sif bash -s <<< "bash && cd ${SCRATCH_DIRECTORY} && python3 run_mechanistic_experiments.py --no-smoketest"
# singularity exec /home/jstastny/numpyro_latest.sif bash -s <<< "bash && cd ${SCRATCH_DIRECTORY} && python3 compute_accuracies.py"
#singularity exec /home/jstastny/numpyro_latest.sif bash -e "python3 ${SCRATCH_DIRECTORY}/run_experiments.py"
# conda list
# python3 ${SCRATCH_DIRECTORY}/run_experiments.py
#sleep 10

# After the job is done we copy our output back to $SLURM_SUBMIT_DIR
cp -r ${SCRATCH_DIRECTORY}/output ${SLURM_SUBMIT_DIR}/${SLURM_JOBID}


# Finish the script
exit 0
