import numpy as np
from conversion_rate import Logistic


class EnviromentUCB(object):

    def __init__(self, arms, n_customers):
        self.arms = arms
        self.n_arms = self.arms.shape[0]
        self.curve = Logistic(self.n_arms/2)
        self.n_customers = n_customers

    def round(self, pulled_arm):
        buyers = round(self.n_customers * self.curve.compute(pulled_arm))
        reward = buyers * (pulled_arm)
        return reward

    def opt_reward(self):
        tmp = []
        for arm in self.arms:
            tmp.append(self.curve.compute(arm) * arm * self.n_customers)
        max_reward = np.max(tmp)
        return max_reward




