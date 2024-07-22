#!/bin/bash
# Specify a partition
#SBATCH --partition=bluemoon
# Setting email preferences
#SBATCH --mail-user=pwelch1@uvm.edu
#SBATCH --mail-type=FAIL,END
# Request nodes 
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=1
# Maximum runtime
#SBATCH --time=30:00:00
# Name of job
#SBATCH --job-name=afpo_ec
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=%x_%j.out

# cd ~/research_code/anthro-scripts

time python3 checkpointing.py