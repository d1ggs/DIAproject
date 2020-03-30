import numpy as np
from prova_ucb.cr_curve import *


class EnviromentUCB(object):

    def __init__(self, arms, n_customers):
        self.arms = arms
        self.n_arms = self.arms.shape[0]
        self.curve = Product1_Season1()
        self.n_customers = n_customers

    def round(self, pulled_arm):
        reward = self.curve.getProbability(pulled_arm) * pulled_arm
        return reward

    def opt_reward(self):
        tmp = []
        for arm in self.arms:
            tmp.append(self.curve.getProbability(arm) * arm)
        max_reward = np.amax(tmp)
        return max_reward





