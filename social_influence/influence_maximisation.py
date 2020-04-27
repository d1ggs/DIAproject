import numpy as np
from itertools import combinations
import multiprocessing
from abc import ABC, abstractmethod

from social_influence.mc_sampling import MonteCarloSampling

class SingleInfluenceLearner(ABC):
    """
    Abstract Class that represent a Influence Maximisation learner for a single Social Network

    Attributes
    --------
    prob_matrix : Edge Activation Probabilities matrix

    n_nodes : number of nodes in the graph

    budget : budget for the the social network
    """
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

        nodes = [d for d in range(0,self.n_nodes)]

        max_influence = 0
        best_seeds = np.zeros(self.n_nodes)
        for i in range(self.budget):
            best_marginal_increase = 0
            step_influence = max_influence

            for n in range(self.n_nodes):
                if best_seeds[n] == 0:
                    #computer marginal increase
                    seeds = np.copy(best_seeds)
                    seeds[n] = 1 #add current node to seeds

                    #computer marginal increase
                    nodes_probabilities = sampler.mc_sampling(seeds, montecarlo_simulations, n_steps_max)
                    influence = np.sum(nodes_probabilities)

                    marginal_increase = influence - step_influence

                    if marginal_increase > best_marginal_increase:
                        best_node = n
                        best_marginal_increase = marginal_increase
                        max_influence = influence

            
            best_seeds[best_node] = 1

        return best_seeds, max_influence

        
class ExactSolutionLearner(SingleInfluenceLearner):
    """
    Exact Solution Influence Maximisation
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

            


