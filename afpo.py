'''
Created on 2022-10-26 10:40:59
@author: caitgrasso

Description: Implementation of Age-Fitness Pareto Optimization (AFPO) for optimization of anthrobot swarms. 
https://dl.acm.org/doi/10.1145/1830483.1830584

NOTE: For AFPO, population size must be significantly large so that not all individuals in the population are on the pareto front. 
'''
import subprocess
from glob import glob
import numpy as np
import copy
import operator
import pickle
import os
import time
import random
import shutil
import matplotlib.pyplot as plt
import constants
from vxa import VXA
from vxd import VXD

class AFPO:

    def __init__(self, random_seed, gens, popsize, maze_level, end_loc, checkpoint_every=50):

        self.seed = random_seed

        # Seed rng 
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.gens = gens
        self.popsize = popsize
        self.checkpoint_every = checkpoint_every
        self.maze_level = maze_level
        self.all_bots_pickles = glob("pickles/*")
        self.next_id = 0
        self.fitness_data = np.zeros(shape=(self.gens+1, self.popsize, 2))
        self.end_loc = end_loc
        # Initialize directories
        os.makedirs('data/', exist_ok=True)
        os.makedirs('history/', exist_ok=True)
        os.makedirs('output/', exist_ok=True)
        os.makedirs('checkpoints/', exist_ok=True)

        self.create_initial_population()

    
    def create_initial_population(self):
        self.pop = [] # population is a list of swarms
        for i in range(self.popsize): # initialize random swarms
            self.pop.append(Swarm(id=self.next_id, bot_composition=np.random.choice(self.all_bots_pickles, constants.SWARM_SIZE), maze_level=self.maze_level,afpo=self))
            self.next_id += 1

    def run(self, continue_from_checkpoint=False, additional_gens=0):
        if continue_from_checkpoint:
            max_gens = self.curr_gen + additional_gens

            
            # Expand fitness data matrix to account for additional gens
            new_fitness_data = np.zeros((max_gens + 1, self.popsize, 2))
            new_fitness_data[0:self.curr_gen,:,:] = self.fitness_data[0:self.curr_gen,:,:]
            self.fitness_data = new_fitness_data

            # self.curr_gen += 1

            self.gens = max_gens

            while self.curr_gen < self.gens + 1:

                self.perform_one_generation()
                if self.curr_gen % self.checkpoint_every == 0:
                    self.save_checkpoint()

                print("GEN: {}".format(self.curr_gen))
                self.print_best(verbose=False)

                self.curr_gen += 1

        else:
            
            self.curr_gen = 0
            
            self.evaluate_generation_zero()
                        
            while self.curr_gen < self.gens + 1: # Evolutionary loop
                
                self.perform_one_generation()
                if self.curr_gen % self.checkpoint_every == 0:
                    self.save_checkpoint()
                
                print("GEN: {}".format(self.curr_gen))
                self.print_best(verbose=False)

                self.curr_gen += 1

        return self.return_best(), self.fitness_data

    def evaluate_generation_zero(self):
        
        # Evaluate individuals in the population
        
        self.evaluate(self.pop)

        for i,swarm in enumerate(self.pop):            
            # Record fitness statistics    
            self.fitness_data[self.curr_gen,i,0] = swarm.fitness
            self.fitness_data[self.curr_gen,i,1] = swarm.age

        print("GEN: {}".format(self.curr_gen))
        self.print_best(verbose=False)

        self.curr_gen+=1

    def perform_one_generation(self):
        self.increase_age()
        children = self.breed()
        children = self.insert_random(children)

        # Evaluate children
        self.evaluate(children)
        for child_swarm in children:
            # Extend population by adding child swarm (extends to popsize*2+1 individuals every generation then gets reduced back to popsize)
            self.pop.append(child_swarm) 

        self.survivor_selection()

        # Record statistics 
        for i, swarm in enumerate(self.pop):
            self.fitness_data[self.curr_gen,i, 0] = swarm.fitness
            self.fitness_data[self.curr_gen,i, 1] = swarm.age

        # self.print_best()

    def increase_age(self):
        for swarm in self.pop:
            swarm.age += 1

    def breed(self):
        children = []
        for i in range(self.popsize):

            # Parent Selection via Tournament Selection (based on fitness only)
            parent = self.tournament_selection()
            
            # Create offspring via mutation
            child = copy.deepcopy(self.pop[parent])
            child.id = self.next_id
            self.next_id += 1
            child.mutate()
            children.append(child)

        return children

    def insert_random(self, children):
        children.append(Swarm(id=self.next_id, bot_composition=np.random.choice(self.all_bots_pickles, constants.SWARM_SIZE), maze_level=self.maze_level, afpo=self))
        self.next_id += 1
        return children

    def tournament_selection(self):
        p1 = np.random.randint(len(self.pop))
        p2 = np.random.randint(len(self.pop))
        while p1 == p2:
            p2 = np.random.randint(len(self.pop))

        if self.pop[p1].fitness > self.pop[p2].fitness:
            return p1
        else:
            return p2

    def survivor_selection(self):

        # Tournament selection based on age and fitness

        # TODO: Check to see if the pareto front size is larger than the population size
        # and increase population size if it is.
        # This shouldn't happen if the population size is large enough but if it does
        # then it would make this function enter an infinite loop -- bad. 

        # Remove dominated individuals until the target population size is reached
        while len(self.pop) > self.popsize:

            # Choose two different individuals from the population
            ind1 = np.random.randint(len(self.pop))
            ind2 = np.random.randint(len(self.pop))
            while ind1 == ind2:
                ind2 = np.random.randint(len(self.pop))

            if self.dominates(ind1, ind2):  # ind1 dominates
                
                # remove ind2 from population and shift following individuals up in list
                for i in range(ind2, len(self.pop)-1):
                    self.pop[i] = self.pop[i+1]
                self.pop.pop() # remove last element from list (because it was shifted up)

            elif self.dominates(ind2, ind1):  # ind2 dominates

                # remove ind1 from population and shift following individuals up in list
                for i in range(ind1, len(self.pop)-1):
                    self.pop[i] = self.pop[i+1]
                self.pop.pop() # remove last element from list (because it was shifted up)

        assert len(self.pop) == self.popsize

    def dominates(self, ind1, ind2):
        # Returns true if ind1 dominates ind2, otherwise false
        if self.pop[ind1].age == self.pop[ind2].age and self.pop[ind1].fitness == self.pop[ind2].fitness:
            return self.pop[ind1].id > self.pop[ind2].id # if equal, return the newer individual

        elif self.pop[ind1].age <= self.pop[ind2].age and self.pop[ind1].fitness >= self.pop[ind2].fitness:
            return True
        else:
            return False

    def evaluate(self, swarms):
        # Number simulations = popsize * constants.N_PERMUTATIONS
        
        # Prepare to simulate by generating vxa/vxds for all swarms
        for swarm in swarms:
            swarm.generate_vxa_vxd()

        # Run set of simulations in voxcraft-sim
        self.submit_batch()

        # Wait for sims to finish (check if history files are complete) 
        self.wait_for_sims_to_finish()

        # Compute fitness based on trajectory data in history files 
        for swarm in swarms:
            swarm.evaluate()
        
        if self.curr_gen != self.gens:
            # Delete all files for this population
            os.system('rm -rf data/swarm*') # clear vxa/vxd files in data except the main base.vxa
            os.system('rm -rf history/*') # clear all history files
            os.system('rm -rf output/*') # clear output

    def submit_batch(self):
        # Iterate through data/ folder and submit all jobs
        data_dirs = glob('data/swarm*/')

        for data_dir in data_dirs:
            history_path = 'history/' + data_dir.split('/')[1] + '.history'
            os.system('sbatch --export=PATH_TO_DATA={},SAVE_PATH={} submit.sh'.format(data_dir, history_path))

            n_jobs = str(subprocess.check_output(['squeue', '-u', 'pwelch1']))
            
            n_jobs = n_jobs.split('\\n')


            while len(n_jobs) > 980:
                time.sleep(15)
                n_jobs = str(subprocess.check_output(['squeue', '-u', 'pwelch1']))
                n_jobs = n_jobs.split('\\n')

    def wait_for_sims_to_finish(self):

        print('Waiting for sims to finish...')
        # TODO: This needs to be checked. Also add some safeguards in case voxcraft crashes or a sim doesn't run for another reason.

        # Check to make sure all sims have started by looking at the number of history files
        all_sims_started = False
        
        n_sims = len(glob('data/swarm*/'))

        while not all_sims_started:
            if len(glob('history/*.history')) == n_sims:
                time.sleep(1) # check in increments of 1 seconds
                all_sims_started = True

        # TODO: Wait for approximate sim time?

        # Check to see if the sims are complete by looking at output files

        finished_sims = []
        while len(finished_sims) != n_sims:
            for out_file in glob('output/*.out'):
                
                f = open(out_file, 'r')
                lines = f.readlines()

                sim_name = lines[0].split('/')[1]

                if sim_name not in finished_sims:

                    for line in lines:
                        if 'real' in line: # real time.. indicates sim has finished
                            finished_sims.append(sim_name)
        
            time.sleep(15) # check in increments of 30 seconds

        print('Sims complete.')

    def save_checkpoint(self):

        filename = 'checkpoints/afpo_maze_level_{}_run{}_{}gens.p'.format(self.maze_level, self.seed, self.curr_gen)
        print('SAVING POPULATION IN: ', filename)

        rng_state = random.getstate()
        np_rng_state = np.random.get_state()

        with open(filename, 'wb') as f:
            pickle.dump([self, rng_state, np_rng_state], f)

    
    def print_population(self, verbose=False):

        for i in range(len(self.pop)):

            self.pop[i].print(verbose=verbose)
        print()

    def print_best(self, verbose):
        best = self.return_best()
        print("BEST SWARM:")
        best.print(verbose=verbose)

    def return_best(self):
        return sorted(self.pop, key=operator.attrgetter('fitness'), reverse=True)[0]


class Swarm:

    def __init__(self, id, bot_composition, maze_level, afpo):
        self.id = id
        self.bot_composition = bot_composition
        self.maze_level = maze_level
        self.read_in_str_cur_indices()
        self.fitness = 0 # to be maximized
        self.age = 0 # gens the individual's genetic material has been in the population
        self.end_loc = afpo.end_loc

    def generate_vxa_vxd(self, permutations=None, record_voxels=False):

        # Load each bot
        bots = []
        for bot in self.bot_composition:
            with open(bot, 'rb') as f:
                cilia, body = pickle.load(f)
            bots.append(self.remove_empty_slices(cilia,body))

        if permutations is None:
            # Keep track of which permutations were done on this bot
            self.permutations = {}
            n_perm = constants.N_PERMUTATIONS
        else:
            self.permutations = permutations
            n_perm = len(permutations)

        for p in range(n_perm):
            # Generate random initial positions/orientations for each bot in swarm

            if permutations is not None:
                init_locations = permutations[p]['init_locations']
                rotations = permutations[p]['rotations']
                # flips = permutations[p]["flips"]
            else:
                self.permutations[p]={}
                
                init_locations = np.random.permutation(self.get_random_initial_locations()) # perumate locations so bot 1 is not always in the bottom left etc..
                self.permutations[p]['init_locations'] = init_locations

                rotations = np.random.permutation(constants.SWARM_SIZE) 
                self.permutations[p]['rotations'] = rotations

                # flips = np.random.randint(2, size=constants.SWARM_SIZE) # 4 random bits
                # self.permutations[p]['flips'] = flips

            # Make body array of 1s and 0s for this permutation
            world = np.zeros((200,200, 7), dtype = int)
            if self.maze_level == "porous_obstructed":
                  for i in range(world.shape[0]):
                    for j in range(world.shape[1]):
                        if i == 0 and j <= 160:
                            world[i,j,0] = 2
                        if i == 160 and j <= 160:
                            world[i, j, 0] = 2
                        if j == 160 and i <= 160:
                            world[i, j, 0] = 2
                        if j == 0 and i <=160:
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 25 and j <= 60):
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 100 and j <= 135):
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 25 and i <= 60): 
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 100 and i <= 135):
                            world[i, j, 0] = 2
            if self.maze_level == "porous_unobstructed":
                  for i in range(world.shape[0]):
                    for j in range(world.shape[1]):
                        if i == 0 and j <= 160:
                            world[i,j,0] = 2
                        if i == 160 and j <= 160:
                            world[i, j, 0] = 2
                        if j == 160 and i <= 160:
                            world[i, j, 0] = 2
                        if j == 0 and i <=160:
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 40 and j <= 60): 
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 100 and j <= 120):
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 40 and i <= 60):
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 100 and i <= 120): 
                            world[i, j, 0] = 2
            if self.maze_level == "nonporous_unobstructed":
                  for i in range(world.shape[0]):
                    for j in range(world.shape[1]):
                        if i == 0 and j <= 160:
                            world[i,j,0] = 2
                        if i == 160 and j <= 160:
                            world[i, j, 0] = 2
                        if j == 160 and i <= 160:
                            world[i, j, 0] = 2
                        if j == 0 and i <=160:
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 40 and j <= 70): 
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 90 and j <= 120):
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 40 and i <= 70):
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 90 and i <= 120):
                            world[i, j, 0] = 2
            if self.maze_level == "nonporous_obstructed":
                  for i in range(world.shape[0]):
                    for j in range(world.shape[1]):
                        if i == 0 and j <= 160:
                            world[i,j,0] = 2
                        if i == 160 and j <= 160:
                            world[i, j, 0] = 2
                        if j == 160 and i <= 160:
                            world[i, j, 0] = 2
                        if j == 0 and i <=160:
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 25 and j <= 70):
                            world[i, j, 0] = 2
                        if (i == 40 or i == 120) and (j >= 90 and j <= 135): 
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 25 and i <= 70):
                            world[i, j, 0] = 2
                        if (j == 40 or j == 120) and (i >= 90 and i <= 135):
                            world[i, j, 0] = 2
            ciliaWorld = np.zeros((world.shape[0], world.shape[1], world.shape[2], 3))

            # Write out body and cilia arrays for a bot to a pickle file
            for i, bot in enumerate(bots):

                cilia, body = bot

                # if flips[i]==1:
                #     cilia = self.flip(cilia)

                cilia = self.rotate_cilia(cilia, k=rotations[i])
        
                # Place bot in the starting area
                x,y = init_locations[i][0], init_locations[i][1]

                world[x:x+body.shape[0], y:y+body.shape[1], :] = body 
                ciliaWorld[x:x+body.shape[0], y:y+body.shape[1], : , :] = cilia
                ciliaWorld[:,:,:,2] = 0 # no cilia forces in the z direction 

            # Write world and world_cilia to vxd
            vxd = VXD()
            vxd.set_vxd_tags(body = world, cilia = ciliaWorld, record_voxels=record_voxels, record_history=True, RecordCoMTraceOfEachVoxelGroupfOfThisMaterial=1) # record COM of first material

            # Save out vxa/vxd file
            data_dir = 'data/swarm{}_perm{}/'.format(self.id, p)
            os.makedirs(data_dir, exist_ok=True)
            vxd.write(data_dir + 'swarm{}_perm{}.vxd'.format(self.id, p))

            # Copy vxa to data_dir computing
            shutil.copy('data/base.vxa', data_dir + 'base.vxa')


    def remove_empty_slices(self, cilia, body):
        '''Hard-coded to remove the empty slices from the body arrays which messes with the cilia rotation'''
        new_body = np.zeros(shape=(7,7,7))
        new_cilia = np.zeros(shape=(7,7,7,3))

        new_body[:,:,:] = body[1:,1:,1:]
        new_cilia[:,:,:,:] = cilia[1:,1:,1:,:]

        return new_cilia,new_body

    def get_random_initial_locations(self):
        '''
        Returns 4 starting xy points within constants.STARTING_AREA_DIM_IN_VOXELS
        xy point indicates bottom right location for the bot

        TODO: this function is hard-coded for 4 bot with bot diameter of 7. Change to accomodate variable swarm/bot sizes.
        '''

        bot1_loc = (np.random.randint(68, 80), np.random.randint(68, 80)) # bottom left starting quadrant
        bot2_loc = (np.random.randint(68, 80), np.random.randint(92, 104)) # bottom left starting quadrant
        bot3_loc = (np.random.randint(92, 104), np.random.randint(68, 80)) # bottom left starting quadrant
        bot4_loc = (np.random.randint(92, 104), np.random.randint(92, 104)) # bottom left starting quadrant

        return [bot1_loc, bot2_loc, bot3_loc, bot4_loc]


    def rotate_cilia(self, cilia, k):
        for i in range(0,k): 
            cilia = self.rotate_90(cilia)
        return cilia

    def rotate_90(self, cilia):
        
        rotated_cilia = np.zeros(shape=cilia.shape)

        temp = np.rot90(cilia, axes=(1,0))

        rotated_cilia[:,:,:,0] = temp[:,:,:,1] * -1
        rotated_cilia[:,:,:,1] = temp[:,:,:,0] 

        return rotated_cilia

    def flip(self, cilia):

        flipped = np.fliplr(cilia)

        flipped[:,:,:,0] = flipped[:,:,:,0] * -1 # rotate force vectors about the y axis

        return flipped
    
    def evaluate(self):

        # Collect history files for this swarm
        history_files = glob('history/swarm{}_perm*.history'.format(self.id))
        
        # Parse history files for each permutation
        all_fitnesses = []
        for history_file in history_files:
            points = self.parse_com_history(history_file)

            # Compute area covered
            swarm_fitness = self.evaluate_h(points)

            all_fitnesses.append(swarm_fitness)

        # Fitness is that of the worst permutation
        self.fitness = np.min(all_fitnesses)
    
    def parse_com_history(self, history_file):
        """Parses history file that records CoM of each voxel group. 

        Args:
            history_file (str): Path to .history file to be parsed.

        Returns:
            dict: Dictionary of trajectories. Key is the bot number and value is an array of size (n_timesteps x 2) where the columns 
                    are the x,y coordinates of the CoM of that bot.
        """    
        f = open(history_file, "r")
        # print(filename)
        
        line = f.readline()

        while line and "real_stepsize" not in line:
            line = f.readline()

        # beginning of trajectory data
        line = f.readline()

        timesteps = []
        trajectories = {}
        # x_coords = []
        # y_coords = []

        while line and "Simulation" not in line and "Stopping" not in line:
            coords = line.split(';')
            coords = coords[:-1] # discard ending parentheses
            # t = int(coords[0].split('}')[0].split("{")[-1])

            for i in range(len(coords)):
                if i==0: # the first set of coords needs to be parsed differently
                    t = int(coords[i].split('}')[0].split("{")[-1])
                    xyz = coords[i].split('}')[-1].split(',')
                else:
                    xyz = coords[i].split(',')
                x = float(xyz[0])
                y = float(xyz[1])
                
                try:
                    trajectories[i].append((x,y))
                except:
                    trajectories[i] = []
                    trajectories[i].append((x,y))

            timesteps.append(t)

            line = f.readline()
        
        f.close()

        # reformat trajectory data

        trajectories_arr = {}

        for i in trajectories:
            points_touples = trajectories[i]

            points_arr = np.reshape(points_touples, newshape=(len(points_touples),2))
        
            trajectories_arr[i] = points_arr


        return trajectories_arr
    def evaluate_h(self,points):

        # Compute the center of the starting points

        # Get initial starting points of all bots
        x_starts = []
        y_starts = []
        for i in points:
            trajectory = points[i]
            x_starts.append(trajectory[0,0])
            y_starts.append(trajectory[0,1])
            
            # Also create array of all coordinates for use in computing the bounding box around the trajectories later
            if i==0:
                all_coords = points[i]
            else:
                all_coords = np.concatenate((all_coords,points[i]))

        center_x = np.mean(x_starts)
        center_y = np.mean(y_starts)

        # Create a boundary of 20*constants.BOT_LENGTH
        min_x = center_x - constants.BOUNDARY_LENGTH_X/2
        max_x = center_x + constants.BOUNDARY_LENGTH_X/2
        min_y = center_y - constants.BOUNDARY_LENGTH_Y/2
        max_y = center_y + constants.BOUNDARY_LENGTH_Y/2

        # Count # unique points
        unique_points = np.unique(all_coords,axis=0)

        # Only keep unique points within the boundary

        # true if in bounds
        x_min_mask = np.reshape(unique_points[:,0]>min_x,newshape=(-1,1))
        x_max_mask = np.reshape(unique_points[:,0]<max_x,newshape=(-1,1))
        y_min_mask = np.reshape(unique_points[:,1]>min_y,newshape=(-1,1))
        y_max_mask = np.reshape(unique_points[:,1]<max_y,newshape=(-1,1))

        mask = np.all(np.concatenate((x_min_mask, x_max_mask, y_min_mask, y_max_mask),axis=1),axis=1)

        unique_points_in_bounds = unique_points[mask]

        return self.fractal_box_count(unique_points_in_bounds,(min_x,min_y,max_x,max_y))

    def fractal_box_count(self, points, boundary):
        # https://francescoturci.net/2016/03/31/box-counting-in-numpy/

        min_x,min_y,max_x,max_y = boundary # unpack tuple

        Ns=[]

        # scales = np.arange(start=2, stop=int(constants.BOUNDARY_LENGTH/constants.MIN_GRID_DIM)) #start with quadrents and go to resolution of voxcraft float
        levels = np.arange(start=1, stop=constants.MAX_LEVEL)

        for level in levels: 

            scale = 2**level

            cell_width = constants.BOUNDARY_LENGTH_X/scale
            cell_height = constants.BOUNDARY_LENGTH_Y/scale

            # H, edges=np.histogramdd(points, bins=(np.linspace(min_x,max_x,constants.BOUNDARY_LENGTH/scale),np.linspace(min_y,max_y,constants.BOUNDARY_LENGTH/scale)))
            H, edges=np.histogramdd(points, bins=(np.linspace(min_x,max_x,num=scale+1),np.linspace(min_y,max_y,num=scale+1)))

            weight = (cell_width*cell_height)/(constants.BOUNDARY_LENGTH_X*constants.BOUNDARY_LENGTH_Y) # David scaling
            Ns.append(np.sum(H>0)*weight)

        # Divide by # of levels to get a value between 0-1
        scaled_box_count = np.sum(Ns)/len(levels) # David scaling

        return scaled_box_count



    def mutate(self):
        # Replace random bot in swarm with new bot
        index_removed = np.random.randint(len(self.bot_composition))
        bot_removed = self.bot_composition[index_removed]
        res_key = bot_removed
        while res_key == bot_removed:
            s = np.random.normal(self.all_str_cur[bot_removed][0],self.all_str_cur[bot_removed][0]/6,1)
            if s > 1: s = 1
            if s < 0: s = 0
            res_key, res_val = min(self.all_str_cur.items(), key=lambda x: abs(s - x[1][0]))

        self.bot_composition[index_removed] = res_key

    def read_in_str_cur_indices(self):
        self.all_str_cur = {}
        f = open("10_7_swarm_data.csv", "r")
        for line in f:
            name_str_cur = line.split(",")
            self.all_str_cur["pickles/"+name_str_cur[0]] = [float(name_str_cur[1]), float(name_str_cur[2])]
        

    def print(self, verbose):
        print("SWARM {}: FITNESS {}: AGE {}".format(self.id, self.fitness, self.age))

        if verbose:
            for i in range(len(self.bot_composition)):
                print("BOT {}: {}".format(i, self.bot_composition[i]))
