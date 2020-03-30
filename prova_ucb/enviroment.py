import numpy as np
from conversion_rate import Logistic


class EnviromentUCB():

    def __init__(self, arms, n_costumers):
        self.arms = arms
        self.n_arms = self.arms.shape[0]
        self.curve = Logistic(self.n_arms/2)
        self.n_costumers = n_costumers

    def round(self, pulled_arm):
        reward = np.random.normal(self.curve.compute(pulled_arm), 0.1)*pulled_arm * self.n_costumers
        return reward

    def opt_reward(self):
        tmp = []
        for arm in self.arms:
            tmp.append(self.curve.compute(arm) * arm * self.n_costumers)
        max_reward = np.amax(tmp)
        return max_reward




