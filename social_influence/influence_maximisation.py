import numpy as np
from itertools import combinations
import multiprocessing

from social_influence.mc_sampling import MonteCarloSampling

class SingleInfluenceLearner(object):
    """
    Attributes
    --------
    prob_matrix : Probability Matrix object

    n_nodes : number of nodes

    budget : budget for the the given social network
    """
    def __init__(self, prob_matrix, n_nodes : int, budget: int):
        super().__init__()
        self.prob_matrix = prob_matrix
        self.n_nodes = n_nodes
        self.budget = budget

    def fit(self, montecarlo_simulations : int, n_steps_max: int):
        """
        Basic exact influence maximization algorithm which enumerates all seeds node given a budget. Returns indexes of best seeds 
        
        Parameters
        ---------
        montecarlo_simulation : number of MonteCarlo Simulations (parameter specified in professor's email)

        n_steps_max : max number of steps in a simulation
        """
        # n = multiprocessing.cpu_count()
        # print("Processors: %d" % n)
        # pool = multiprocessing.Pool()

        sampler = MonteCarloSampling(self.prob_matrix)

        nodes = [d for d in range(0,self.n_nodes)]
        #seeds_combinations = list(combinations(nodes, self.budget)) #this take too much memory
        

        max_influence = 0
        best_seeds = np.zeros(self.n_nodes)
        sum_simulations = np.zeros(montecarlo_simulations) 

        for combination in combinations(nodes, self.budget): #enumerate all possible seeds given a budget
            seeds = np.zeros(self.n_nodes)
            #print("Current Seeds: [%s]" % (','.join(str(n) for n in combination)))
            seeds[list(combination)] = 1
            
            #each row contains the nodes probalities from 1 to n=montecarlo_simulations
            nodes_probabilities_for_simulation = sampler.mc_sampling(seeds, montecarlo_simulations, n_steps_max)

            #the best seed is the one where the sum of probabilities is the highest
            total_sum = np.sum(nodes_probabilities_for_simulation, axis= 1) 

            if (total_sum.max() > max_influence): 
                max_influence = total_sum.max()
                sum_simulations = total_sum #(keep track of all simulation to print the approximation error as specified in point 2) of the project
                best_seeds = seeds
            
        #print("Best Seeds: [%s] Result: %.2f" % (','.join(str(n) for n in seeds), max_influence))
        
        return best_seeds, sum_simulations

            


