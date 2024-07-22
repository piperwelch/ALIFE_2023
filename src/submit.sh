#!/bin/bash
# specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=4
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=16G
# Maximum runtime of 1 min
#SBATCH --time=00:01:00
# Name of this job
#SBATCH --job-name=grey_goo
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=output/%x_%j.out

spack load gcc@7.3.0
spack load cmake@3.17.1%gcc@7.3.0 arch=linux-rhel7-haswell
spack load boost@1.66.0
spack load cuda

# change to the directory where script is submitted (should also be where the voxcraft executables are located)
cd ${SLURM_SUBMIT_DIR}

echo $PATH_TO_DATA
echo $SAVE_PATH

time ./voxcraft-sim -i ${PATH_TO_DATA} > ${SAVE_PATH}

