import numpy as np

from tutorials.pricing.learner import Learner


class GreedyLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        # Pull each arm at least one time
        if self.t < self.n_arms:
            return self.t

        # Pull the arm that has the highest expected value and perform random tie breaking if needed
        idx = np.random.choice(np.argwhere(self.expected_rewards == np.max(self.expected_rewards))[0])
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        # Update the expected rewards with the incremental mean
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t
