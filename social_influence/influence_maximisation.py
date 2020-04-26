import numpy as np
from itertools import combinations
import multiprocessing
from abc import ABC, abstractmethod

from social_influence.mc_sampling import MonteCarloSampling

class SingleInfluenceLearner(ABC):
    def __init__(self, prob_matrix, n_nodes : int, budget: int):
        self.prob_matrix = prob_matrix
        self.n_nodes = n_nodes
        self.budget = budget
    
    @abstractmethod
    def fit(self):
        pass

class GreedyLearner(SingleInfluenceLearner):

    def fit(self, montecarlo_simulations : int, n_steps_max: int):

        sampler = MonteCarloSampling(self.prob_matrix)

        
        

        
class ExactSolutionLearner(SingleInfluenceLearner):
    """
    Exact Solution Influence Maximisation

    Attributes
    --------
    prob_matrix : Probability Matrix object

    n_nodes : number of nodes

    budget : budget for the the given social network
    """
    
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
        

        for combination in combinations(nodes, self.budget): #enumerate all possible seeds given a budget
            seeds = np.zeros(self.n_nodes)
            #print("Current Seeds: [%s]" % (','.join(str(n) for n in combination)))
            seeds[list(combination)] = 1
            
            nodes_probabilities = sampler.mc_sampling(seeds, montecarlo_simulations, n_steps_max)

            #the best seed is the one where the sum of probabilities is the highest
            influence = np.sum(nodes_probabilities) 

            if (influence> max_influence): 
                max_influence = influence
                best_seeds = seeds
            
        #print("Best Seeds: [%s] Result: %.2f" % (','.join(str(n) for n in seeds), max_influence))
        
        return best_seeds, max_influence

            


