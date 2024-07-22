#!/bin/bash
# Specify a partition 
#SBATCH --partition=dggpu
# Request nodes 
#SBATCH --nodes=1
# Request some processor cores 
#SBATCH --ntasks=4
# Request GPUs 
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=25G
# Maximum runtime of 2 hours
#SBATCH --time=00:10:00

# spack load cuda
# spack load gcc@7.3.0
# spack load cmake@3.17.1
# spack load boost@1.66.0
# spack load py-numpy@1.18.4

# ./voxcraft-sim -i $1 -o results.xml -f > swarm_results/swarm_id_$2_gen_$3_perm_$4.history
python3 eval_sims.py $1