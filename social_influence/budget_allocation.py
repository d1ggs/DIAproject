import numpy as np
from scipy.optimize import linprog
from social_influence.influence_maximisation import GreedyLearner

class GreedyBudgetAllocation:

    def __init__(self, social1, social2, social3, budget, mc_simulations, n_steps_montecarlo):

        """
        In the main we need to pass 3 social media objects through the main
        Parameters
        ----------
        social1
        social2
        social3
        """
        self.social1_learner = GreedyLearner(social1.get_matrix(), social1.get_n_nodes())
        self.social2_learner = GreedyLearner(social2.get_matrix(), social2.get_n_nodes())
        self.social3_learner = GreedyLearner(social3.get_matrix(), social3.get_n_nodes())
        self.budget = budget
        self.mc_simulations = mc_simulations
        self.n_steps_montecarlo = n_steps_montecarlo

    def joint_influence_calculation(self, budget):
        seeds1, influence1 = self.social1_learner.fit(self.mc_simulations, self.n_steps_montecarlo, budget[0])
        seeds2, influence2 = self.social1_learner.fit(self.mc_simulations, self.n_steps_montecarlo, budget[1])
        seeds3, influence3 = self.social1_learner.fit(self.mc_simulations, self.n_steps_montecarlo, budget[2])
        return influence1 + influence2 + influence3

    def joint_influence_maximization(self):
        #TODO discrete optimization che porcodio maiale scipy non lo fa

        return 1







        




