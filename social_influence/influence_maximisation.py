from social_influence.mc_sampling import MonteCarloSampling
    
import numpy as np
from itertools import combinations

class SingleInfluenceLearner(object):
    """
    Attributes
    --------
    sampler : MonteCarloSampling object

    n_nodes : number of nodes

    budget : budget for the the given social network
    """
    def __init__(self, sampler : MonteCarloSampling, n_nodes : int, budget: int):
        super().__init__()
        self.sampler = sampler
        self.n_nodes = n_nodes
        self.budget = budget

    def fit(self):
        """
        Basic exact influence maximization algorithm which enumerates all seeds node given a budget. Returns indeces of best seeds 
        """
        seeds_combinations = list(combinations([d for d in range(0,self.n_nodes)], self.budget))
        n_episodes = [2] #parameter 
        n_steps_max = 5 #parameter
        max_influence = 0
        seeds = []
        for i, combination in enumerate(seeds_combinations): #enumerate all possible seeds given a budget
            seeds = np.zeros(self.n_nodes)
            seeds[[combination]] = 1
            for n in n_episodes:
                nodes_probabilities = self.sampler.mc_sampling(seeds, n_episodes[0], n_steps_max)
                total_sum = np.sum(nodes_probabilities)
                if (total_sum > max_influence):
                    max_influence = total_sum
                    seeds = combination
                #print("Seeds: [%s] ] Result: %f" % (','.join(str(n) for n in combination), total_sum))
        
        print("Best Seeds: [%s] Result: %.2f" % (','.join(str(n) for n in seeds), max_influence))
        return seeds
            


