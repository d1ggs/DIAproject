import numpy as np


class Learner(object):
    """Class implementing a generic learner"""
    def __init__(self, prices:list):
        """:param n_arms: the number of arms of the MAB setting"""

        self.prices = prices
        self.n_arms = len(prices)
        self.t = 0  # Represents the current round
        self.rewards_per_arm = [[] for i in range(self.n_arms)]  # Stores the collected rewards at each round for each arm
        self.collected_rewards = np.array([])
        self.mean_reward_per_arm = np.zeros(self.n_arms)
        self.pull_count = np.zeros(self.n_arms)

    def update_observations(self, pulled_arm: int, reward: float):
        """
        Updates the history of collected rewards for each arm and the global history

        :param reward: the reward collected after pulling an arm at the current round
        :param pulled_arm: the index of the arm that has been pulled at the current round
        """

        # Store the reward
        self.rewards_per_arm[pulled_arm].append(reward)
        self.pull_count[pulled_arm] += 1
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.mean_reward_per_arm[pulled_arm] += (reward - self.mean_reward_per_arm[pulled_arm]) / self.pull_count[pulled_arm]

