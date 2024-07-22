'''
Created on 2022-10-26 17:58:58
@author: caitgrasso

Description: Driver script to continue evolutionary run of AFPO from a checkpoint.
'''
import random 
import numpy as np
import pickle
import os

from afpo import AFPO
from vxa import VXA

def continue_from_checkpoint(checkpoint_file, additional_gens):

    # Remake vxa and save in the data folder to be copied later
    vxa = VXA(SimTime = 5, EnableCilia=1, EnableCollision=1, RecordVoxel=1, RecordFixedVoxels=1)
    vxa.add_material(RGBA=(255,0,0, 255), E=9e4, Cilia=0.0006, LockZ = 1)
    vxa.add_material(RGBA=(0,0,0, 255), E=9e4, LockZ = 1, isMeasured=1, isFixed=1) # passive

    vxa.write('data/base.vxa')
    
    # Load rng state and afpo state
    with open(checkpoint_file, 'rb') as f:
        afpo, rng_state, np_rng_state = pickle.load(f)

    # Reseed the random number generator
    random.setstate(rng_state)
    np.random.set_state(np_rng_state)

    best, fitness_data = afpo.run(continue_from_checkpoint=True, additional_gens=additional_gens)

    best.print(verbose=True)
    print(fitness_data)

    # pickle out best bot and save csv of fitness data
    os.makedirs('results/', exist_ok=True)
    filename = 'afpo_seed{}_gens{}_popsize{}'.format(afpo.seed, afpo.gens, afpo.popsize)
    with open('results/'+filename+'_best.p','wb') as f:
        pickle.dump(best, f)
    np.savetxt('results/'+filename+'_fitness_data.csv', fitness_data, delimiter=',')

seed = 0
gen = 0 
checkpoint_filename = f'checkpoints/afpo_maze_level_nonporous_obstructed_run{seed}_{gen}gens.p'

continue_from_checkpoint(checkpoint_filename, additional_gens=24)
