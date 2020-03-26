import numpy as np


class Learner(object):
    """Class implementing a generic learner"""
    def __init__(self, n_arms: int):
        """:param n_arms: the number of arms of the MAB setting"""

        self.n_arms = n_arms
        self.t = 0  # Represents the current round
        self.rewards_per_arm = x = [[] for i in range(n_arms)]  # Stores the collected rewards at each round for each arm
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm: int, reward: float):
        """
        Updates the history of collected rewards for each arm and the global history

        :param reward: the reward collected after pulling an arm at the current round
        :param pulled_arm: the index of the arm that has been pulled at the current round
        """

        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

