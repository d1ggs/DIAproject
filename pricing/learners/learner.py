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

    def update_observations(self, pulled_arm: int, reward: float):
        """
        Updates the history of collected rewards for each arm and the global history

        :param reward: the reward collected after pulling an arm at the current round
        :param pulled_arm: the index of the arm that has been pulled at the current round
        """

        # Store the reward
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.t = 0

    def get_last_best_price(self):
        """
        The method computes the price which in expectation provides the largest value if proposed to all users
        :return: the value of the best price
        """

        if self.t == 0:
            return np.random.choice(self.prices)

        expected_reward = np.mean(self.rewards_per_arm, axis=1)
        for i in range(len(expected_reward)):
            expected_reward[i] *= self.prices[i]

        best_price_index = np.argmax(expected_reward)

        return self.prices[best_price_index]

