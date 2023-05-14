parallelize = True
POOL_NUM = 16



# MAP_NUM = 10
MAP_SIZE = 10000

OBS_NUM = 5000
OBS_A_MIN = 20
OSP_A_MAX = 30
OBS_B_MIN = 20
OSP_B_MAX = 30

# START = (int(MAP_SIZE/100), int(MAP_SIZE/100))
# GOAL = (int(MAP_SIZE - MAP_SIZE/100), int(MAP_SIZE - MAP_SIZE/100))

QUAD_TREE_MAX_LEVEL = 9
OBSTACLE_PERCENT = 0.99

LENGTH_COEF = 1
TIME_COEF = 1
SMOOTH_COEF = 100
COLLISION_COEF = 100

SUB_PROB_SAMPLE_FACTOR = 20
INSERT_MOVE_DIST = 500


start_goal_dict = {'Ellipse': ((100, 100), (9900, 100)),
                   'Enigma': ((9500, 1500), (500, 9500)),
                   'Labyrinth': ((2000, 500), (8000, 9500)),
                   'LakeShore': ((2000, 1000), (8000, 9000)),
                   'PrimevalIsles': ((9500, 500), (400, 9600)),
                   'dense': ((100, 100), (9900, 9900)),
                   'MultiObj': ((500, 500), (9500, 500))}

gen_num_dict = {'Ellipse': 3,
                'Enigma': 3,
                'Labyrinth': 3,
                'LakeShore': 4,
                'PrimevalIsles': 4,
                'dense': 4,
                'MultiObj': 2}


# parameters of the main GA
num_parents_mating = 16
sol_per_pop = 32

keep_parents = -1
elitism = True
mutate_parent = True

selection_type = "tournament"
K_tournament = 2

# selection_type = "roulette"
# K_tournament = None

crossover_probability = 1
mutation_probability = 1




# parameters of the sub-problem GA
num_generations_sub = 200

num_parents_mating_sub = 3
sol_per_pop_sub = 9

keep_parents_sub = -1
elitism_sub = True

selection_type_sub = "tournament"
K_tournament_sub = 2

crossover_probability_sub = 1
mutation_probability_sub = 1


# parameters of benchmark models
num_generations_bench = 800
num_parents_mating_bench = 160
sol_per_pop_bench = 320
