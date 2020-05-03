import numpy as np
from social_influence.influence_maximisation import

class GreedyBudgetAllocation:

    def __init__(self, social1, social2, social3, budget):

        """
        In the main we need to pass 3 social media objects through the main
        Parameters
        ----------
        social1
        social2
        social3
        """
        self.social1_matrix = social1.matrix
        self.social2_matrix = social2.matrix
        self.social3_matrix = social3.matrix
        self.budget = budget

    def joint_influence_maximization(self):

        




