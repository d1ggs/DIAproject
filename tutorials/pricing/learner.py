import numpy as np


class Learner(object):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0  # Represents the current round
        self.rewards_per_arm = x = [[] for i in range(n_arms)]  # Stores the collected rewards at each round for each arm
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

