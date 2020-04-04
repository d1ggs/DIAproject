from pricing import Learner
import numpy as np


class TSLearner(Learner):
    def __init__(self, n_arms: int, prices: list):
        assert n_arms == len(prices), "Number of prices different from the number of arms"
        super().__init__(n_arms)

        # self.prices = np.random.randint(0, 100, self.n_arms)
        # print(self.prices)
        self.prices = prices
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        samples = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        vec = []
        for i in range(len(samples)):
            vec.append(samples[i] * self.prices[i])
        idx = np.argmax(np.array(vec))
        return idx

    def update(self, pulled_arm, reward):

        self.t += 1

        # Update the observations list
        self.update_observations(pulled_arm, reward)

        # Update the beta distributions for the pulled arm
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1 - reward

