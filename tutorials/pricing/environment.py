import numpy as np


class Environment(object):
    def __init__(self, n_arms: int, probabilities: np.ndarray):
        """
        This class models a MAB setting, storing probabilities of success for each arm and providing stochastic rewards
        for each arm

        :param n_arms, the number of arms of the MAB setting
        :param probabilities, the success probability for each Bernoulli distribution associated to an arm
        """

        super().__init__()
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm: int):
        """Computes the reward for the pulled arms, drawing from a binomial distribution"""
        reward = np.random.binomial(1, self.probabilities[pulled_arm])  # the first parameter is 1 because it is a Bernoulli
        return reward
