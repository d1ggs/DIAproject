from tutorials.pricing.learner import Learner
import numpy as np


class ThompsonSamplingLearner(Learner):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1

        # Update the observations list
        self.update_observations(pulled_arm, reward)

        # Update the beta distributions for the pulled arm
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1 - reward

