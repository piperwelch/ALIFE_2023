# Swarm parameters
SWARM_SIZE = 4
N_PERMUTATIONS = 6
BOT_DIAM = 7
STARTING_AREA_DIM = BOT_DIAM*4+8 #(7 bots, 4 starting quadrants, plus 8 in case one bot starts on the far edge)

# For fitness function calculations
LATTICE_DIM = 0.01 # 1 voxel is LATTICE_DIM meters
BOT_LENGTH = 8*LATTICE_DIM # diameter in voxels * lattice_dim
BOUNDARY_LENGTH_X = 320*LATTICE_DIM #change this based on environment size 
BOUNDARY_LENGTH_Y = 320*LATTICE_DIM #change this based on environment size 
MAX_LEVEL = 13
