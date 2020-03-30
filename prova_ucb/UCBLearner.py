import numpy as np
from tutorials.pricing.learners.learner import Learner


class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.upper_bounds = np.ones(n_arms) * np.inf

    def pull_arm(self):
        idx = np.argmax(self.upper_bounds)
        return idx

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.t += 1
        x = np.mean(self.rewards_per_arm[pulled_arm])
        self.upper_bounds[pulled_arm] = x + np.sqrt((2 * np.log(self.t)) / len(self.rewards_per_arm[pulled_arm]))



