from quad_tree import *
import pygad
import random
import time
import numpy as np
from multiprocessing import Pool, Manager


manager = Manager() 
seg_cost_history = manager.dict()

islands, islands_ranges, islands_areas = [], [], []

class GA_focus():
    def __init__(self, 
                 num_generations, 
                 num_parents_mating, 
                 fitness_func,
                 pop_size=None,
                 initial_population=None,
                 pick_corners=False, 
                 elitism=True,
                 keep_parents=-1,
                 selection_type="roulette",
                 K_tournament=2,
                 crossover_type=None,
                 crossover_probability=None,
                 mutation_type=None,
                 mutation_probability=None,
                 on_start=None,
                 on_fitness=None,
                 on_parents=None,
                 on_crossover=None,
                 on_mutation=None,
                 # on_generation=None,
                 # on_stop=None,
                 delay_after_gen=0.0,
                 save_best_solutions=True,
                 # save_solutions=False,
                 # stop_criteria=None,
                 # parallel_processing=None,
                 random_seed=None,
                 free_blocks=None,
                 start_pos=None,
                 goal_pos=None,
                 islands=None,
                 islands_areas=None,
                 islands_ranges=None,
                 segment_cost_func=None,
                 mutate_parent=False):
        
        self.random_seed = random_seed
        if not random_seed is None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            
        self.pick_corners = pick_corners
            
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.fitness_func = fitness_func
        
        self.elitism = elitism
        self.keep_parents = keep_parents
        self.selection_type = selection_type
        self.K_tournament = K_tournament
        self.last_generation_parents = []
        self.last_generation_parents_indices = []
        
        
        if callable(crossover_type):
            self.crossover = crossover_type
        else:
            raise("Crossover func not defined!")
            
        self.crossover_probability = crossover_probability
        
        if callable(mutation_type):
            self.mutation = mutation_type
        else:
            raise("Mutation func not defined!")
        
        self.mutation_probability = mutation_probability
        
        self.on_start = on_start
        self.on_fitness = on_fitness
        self.on_parents = on_parents
        self.on_crossover = on_crossover
        self.on_mutation = on_mutation
        
        self.save_best_solutions = save_best_solutions
        self.best_solutions = None
        self.best_solutions_fitness = None
        
        self.generations_completed = 0
        self.delay_after_gen = delay_after_gen
        
        if free_blocks is None:
            raise("Free blocks not defined!")
        self.free_blocks = free_blocks
        
        if not initial_population is None:
            self.population=initial_population
            self.population_size = len(self.population)
        else:
            self.population_size = pop_size
            if pop_size < 100:
                self.initialize_population_plus_plus()
            else:
                self.initialize_population()
     
        if self.keep_parents == -1:
            self.num_offspring = len(self.population) - self.num_parents_mating
        else:
            self.num_offspring = len(self.population) - min(self.keep_parents, self.num_parents_mating)
        
        if self.elitism:
            self.num_offspring = self.num_offspring - 1
        self.last_generation_elitism = None
        
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        self.islands = islands
        self.islands_areas = islands_areas
        self.islands_ranges = islands_ranges
        
        self.segment_cost_func = segment_cost_func
        self.mutate_parent = mutate_parent
    
    def initialize(self, init_population=None):
        if init_population is None:
#             self.initialize_population()
            self.initialize_population_plus_plus()
        else:
            self.population = init_population
            self.population_size = len(self.population)
        self.generations_completed = 0
        self.last_generation_parents = []
        self.last_generation_parents_indices = []
    
    def initialize_population(self, num_wp=1):
        initial_population = []
        for i in range(self.population_size):
            init_path = []
            for i in range(num_wp):
                x, y = pick_random_wp(self.free_blocks)
                init_path.append((x, y))
            initial_population.append(init_path)
        self.population = initial_population
        
    def initialize_population_plus_plus(self):
        # initialize the population with the method in kmeans++
        print("initialize with ++")
        samples = []
        for i in range(10 * self.population_size):
            x, y = pick_random_wp(self.free_blocks)
            samples.append((x, y))
        
        initial_population = []
        for i in range(self.population_size):
            if not initial_population:
                idx = np.random.choice(range(len(samples)))
                initial_population.append([samples[idx]])
            else:
                best_dist = 0
                best_point = None
                for p0 in samples:
                    if [p0] in initial_population:
                        continue
                    x0, y0 = p0
                    dist = 1e10
                    for individual in initial_population:
                        p1 = individual[0]
                        x1, y1 = p1
                        dist = min(dist, euclidean(x0, y0, x1, y1))
                    if dist > best_dist:
                        best_dist = dist
                        best_point = p0
                initial_population.append([best_point])
        
        self.population = initial_population
        
    def set_start_goal(self, start_pos, goal_pos):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
    
    def best_solution(self, pop_fitness=None):
        if pop_fitness is None:
            pop_fitness = [self.fitness_func(solution, idx, self.start_pos, self.goal_pos) for idx, solution in enumerate(self.population)]
        best_solution_fitness = max(pop_fitness)
        best_match_idx = pop_fitness.index(best_solution_fitness)
        best_solution = self.population[best_match_idx]
        return best_solution, best_solution_fitness, best_match_idx

    
    def cal_pop_fitness(self):
#         new_segs = set()
        
#         # find all new segments to be evaluated
#         for solution in self.population:
#             segments = []
#             if len(solution) == 0:
#                 segments = [(self.start_pos, self.goal_pos)]
#             else:
#                 segments.append((self.start_pos, solution[0]))
#                 if len(solution) >= 2:
#                     for i in range(0, len(solution)-1):
#                         segments.append((solution[i], solution[i+1]))
#                 segments.append((solution[-1], self.goal_pos))
                
#             for seg in segments:
#                 if seg not in seg_cost_history:
#                     new_segs.add(seg)
        
#         # calculate the cost of new segments and save the cost in seg_cost_history
# #         print(list(new_segs))
#         with Pool(16) as p:
#             seg_costs = p.starmap(self.segment_cost_func, list(new_segs))
        
#         for A, B in new_segs:
#             self.segment_cost_func(A, B)
            
        pop_fitness = [self.fitness_func(solution, idx, self.start_pos, self.goal_pos) for idx, solution in enumerate(self.population)]
        return pop_fitness
    
    def select_parents(self, fitness, num_parents):
        if self.selection_type == "roulette":
            return self.roulette_wheel_selection(fitness, num_parents)
        else:
            return self.tournament_selection(fitness, list(range(len(fitness))), num_parents)
    
    def tournament_selection(self, fitness, indices, num_parents):
        parents = []
        parents_indices = []

        for parent_num in range(num_parents):
            rand_indices = np.random.choice(indices, size=self.K_tournament)
            K_fitnesses = [fitness[idx] for idx in rand_indices]
#             selected_parent_idx = np.where(K_fitnesses == np.max(K_fitnesses))[0][0]
            selected_parent_idx = K_fitnesses.index(max(K_fitnesses))
            parents_indices.append(rand_indices[selected_parent_idx])
            parents.append(self.population[rand_indices[selected_parent_idx]].copy())

        return parents, parents_indices
    
    def roulette_wheel_selection(self, fitness, num_parents):
        fitness = np.array(fitness)
        fitness_sum = np.sum(fitness)
        if fitness_sum == 0:
            raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
        if fitness_sum < 0:
            raise ZeroDivisionError("Cannot proceed because the sum of fitness values is less than zero.")
        probs = fitness / fitness_sum
        probs_start = np.zeros(probs.shape, dtype=np.float) # An array holding the start values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape, dtype=np.float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        parents = []
        parents_indices = []

        for parent_num in range(num_parents):
            rand_prob = np.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents.append(self.population[idx].copy())
                    parents_indices.append(idx)
                    break
        return parents, parents_indices

    
    
    def run(self):

        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """
        
        if self.save_best_solutions:
            self.best_solutions = []
        self.best_solutions_fitness = []

        if not (self.on_start is None):
            self.on_start(self)

        stop_run = False

        # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
        self.last_generation_fitness = self.cal_pop_fitness()

        best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)
        self.last_generation_elitism = best_solution.copy()
        
        # Appending the best solution in the initial population to the best_solutions list.
        if self.save_best_solutions:
            self.best_solutions.append(best_solution)

        for generation in range(self.num_generations):
#             if not self.random_seed is None:
#                 self.random_seed += 1000
#                 np.random.seed(self.random_seed)
#                 random.seed(self.random_seed)
            
            if not (self.on_fitness is None):
                self.on_fitness(self, self.last_generation_fitness)

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(best_solution_fitness)

            # Selecting the best parents in the population for mating.
            self.last_generation_parents, self.last_generation_parents_indices = self.select_parents(self.last_generation_fitness, num_parents=self.num_parents_mating)
                
            if not (self.on_parents is None):
                self.on_parents(self, self.last_generation_parents)

            # Crossover
            self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                                      self.num_offspring,
                                                                      self)
            if not (self.on_crossover is None):
                self.on_crossover(self, self.last_generation_offspring_crossover)

            # Mutation
            self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover, self.mutation_probability, self.free_blocks, self)
            if not (self.on_mutation is None):
                self.on_mutation(self, self.last_generation_offspring_mutation)
            
            
            if (self.keep_parents < 0):
                if self.mutate_parent:
                    mutated_parents = self.mutation(self.last_generation_parents, self.mutation_probability, self.free_blocks, self)
                    self.population = self.last_generation_offspring_mutation + mutated_parents
                else:
                    self.population = self.last_generation_offspring_mutation + self.last_generation_parents
            else:
                self.last_generation_parents.sort(key=lambda x : -self.fitness_func(x, 0, self.start_pos, self.goal_pos))
                self.population = self.last_generation_offspring_mutation + self.last_generation_parents[:self.keep_parents]
            
            if self.elitism:
                if self.mutate_parent:
                    mutated_elitism = self.mutation([self.last_generation_elitism.copy()], self.mutation_probability, self.free_blocks, self)
                    self.population = self.population + mutated_elitism
                else:
                    self.population.append(self.last_generation_elitism.copy())
            
            self.generations_completed = generation + 1 # The generations_completed attribute holds the number of the last completed generation.

            self.previous_generation_fitness = self.last_generation_fitness.copy()
            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()

            best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)
            self.last_generation_elitism = best_solution
            
            # Appending the best solution in the current generation to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)

            
                
#             # If the callback_generation attribute is not None, then cal the callback function after the generation.
#             if not (self.on_generation is None):
#                 r = self.on_generation(self)
#                 if type(r) is str and r.lower() == "stop":
#                     # Before aborting the loop, save the fitness value of the best solution.
#                     _, best_solution_fitness, _ = self.best_solution()
#                     self.best_solutions_fitness.append(best_solution_fitness)
#                     break

#             if not self.stop_criteria is None:
#                 for criterion in self.stop_criteria:
#                     if criterion[0] == "reach":
#                         if max(self.last_generation_fitness) >= criterion[1]:
#                             stop_run = True
#                             break
#                     elif criterion[0] == "saturate":
#                         criterion[1] = int(criterion[1])
#                         if (self.generations_completed >= criterion[1]):
#                             if (self.best_solutions_fitness[self.generations_completed - criterion[1]] - self.best_solutions_fitness[self.generations_completed - 1]) == 0:
#                                 stop_run = True
#                                 break

#             if stop_run:
#                 break

            time.sleep(self.delay_after_gen)

#         # Save the fitness of the last generation.
#         if self.save_solutions:
#             # self.solutions.extend(self.population.copy())
#             # population_as_list = self.population.copy()
#             population_as_list = [item.copy() for item in self.population]
#             self.solutions.extend(population_as_list)

#             self.solutions_fitness.extend(self.last_generation_fitness)

        # Save the fitness value of the best solution.
        _, best_solution_fitness, _ = self.best_solution(pop_fitness=self.last_generation_fitness)
        self.best_solutions_fitness.append(best_solution_fitness)

        # self.best_solution_generation = np.where(np.array(self.best_solutions_fitness) == np.max(np.array(self.best_solutions_fitness)))[0][0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True # Set to True only after the run() method completes gracefully.

#         if not (self.on_stop is None):
#             self.on_stop(self, self.last_generation_fitness)
    
    
def my_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size:
        if ga_instance.crossover_probability is None:
            parent1 = parents[idx % len(parents)].copy()
            parent2 = parents[(idx + 1) % len(parents)].copy()
        else:
            probs = np.random.random(size=len(parents))
            indices = np.where(probs <= ga_instance.crossover_probability)[0]

            # If no parent satisfied the probability, no crossover is applied and a parent is selected.
            if len(indices) == 0:
                offspring.append(parents[idx % len(parents)].copy())
                continue
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = np.random.choice(list(set(indices)), size=2)
                parent1_idx = indices[0]
                parent2_idx = indices[1]
            parent1 = parents[parent1_idx].copy()
            parent2 = parents[parent2_idx].copy()

        shared = set(parent1).intersection(parent2)
        
        if len(parent1) > 0 and len(parent2) > 0:
            path1 = [ga_instance.start_pos] + parent1.copy() + [ga_instance.goal_pos]
            path2 = [ga_instance.start_pos] + parent2.copy() + [ga_instance.goal_pos]

            length1 = []
            length2 = []
            
            for i in range(len(path1)-1):
                x1, y1 = path1[i]
                x2, y2 = path1[i+1]
                if length1:
                    length1.append(length1[-1] + euclidean(x1, y1, x2, y2))
                else:
                    length1.append(euclidean(x1, y1, x2, y2))
            
            for i in range(len(path2)-1):
                x1, y1 = path2[i]
                x2, y2 = path2[i+1]
                if length2:
                    length2.append(length2[-1] + euclidean(x1, y1, x2, y2))
                else:
                    length2.append(euclidean(x1, y1, x2, y2))
            

            p1p = np.random.choice(range(len(parent1)+1))
            p2p = 0
            for i in range(len(length2)):
                if length2[i] / length2[-1] >= length1[p1p] / length1[-1]:
                    p2p = i
                    break
            
            child = parent1[:p1p] + parent2[p2p:]
        
        else:
            p1p = np.random.choice(range(len(parent1)+1))
            p2p = np.random.choice(range(len(parent2)+1))

            child = parent1[:p1p] + parent2[p2p:]
        
        child = remove_duplicate_point(child, ga_instance)
        offspring.append(child)

        idx += 1
    
#     for i in range(len(offspring)):
#         offspring[i] = remove_round(offspring[i])

    return offspring

def remove_duplicate_point(path, ga_instance):
    new_path = []
    for p in path:
        if p not in new_path and p != ga_instance.start_pos and p != ga_instance.goal_pos:
            new_path.append(p)
    return new_path


def remove_round(path):
    result = path.copy()
    for i in range(len(result) - 1):
        for j in range(len(result) - 1, i, -1):
            if result[i] == result[j]:
                return result[:i] + result[j:]
    return result
    

# def my_mutation_raw(offspring, mutation_probability, free_blocks, ga_instance):
#     mutation_types = ["move", "delete", "insert"]
    
#     # pick offspring to mutate
#     mutated_idx = random.choices(range(len(offspring)), k=int(len(offspring) * mutation_probability))
    
#     for chromosome_idx in mutated_idx:
#         mutation_weights = [MOVE_WEIGHT, DELETE_WEIGHT, 2]
#         # print(f"before mutation: {offspring[chromosome_idx]}")
#         wp_num = len(offspring[chromosome_idx])
        
#         if wp_num == 0:
#             mutation_weights[0] = 0
#             mutation_weights[1] = 0
#         mtype = random.choices(mutation_types, mutation_weights)[0]
        
#         if mtype == "move":
#             random_wp_idx = random.choice(range(wp_num))
#             if ga_instance.pick_corners:
#                 xm, ym = pick_random_corner_wp(free_blocks)
#             else:
#                 xm, ym = pick_random_wp(free_blocks)
#             offspring[chromosome_idx][random_wp_idx] = (xm, ym)
            
#         elif mtype == "delete":
#             random_wp_idx = random.choice(range(wp_num))
#             offspring[chromosome_idx].pop(random_wp_idx)
            
#         else:  # if mtype == "insert"
#             random_wp_idx = random.choice(range(wp_num + 1))
#             if ga_instance.pick_corners:
#                 xi, yi = pick_random_corner_wp(free_blocks)
#             else:
#                 xi, yi = pick_random_wp(free_blocks)
#             offspring[chromosome_idx].insert(random_wp_idx, (xi, yi))
            
#         # print(f"after mutation: {offspring[chromosome_idx]}")
        
#     return offspring

def draw_solution(solution, img, color=[0, 255, 0], start=None, goal=None):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if start is None:
        start = START
    if goal is None:
        goal = GOAL
    waypoints = [start] + solution + [goal]
    for i in range(len(waypoints)-1):
        img = cv2.line(img, waypoints[i], waypoints[i+1], color, 5)
    
    return img

# Sample of using user-defined crossover/mutation function
# ga_instance = pygad.GA(num_generations=10,
#                        sol_per_pop=5,
#                        num_parents_mating=2,
#                        num_genes=len(equation_inputs),
#                        fitness_func=fitness_func,
#                        crossover_type=crossover_func,
#                        mutation_type=mutation_func)
