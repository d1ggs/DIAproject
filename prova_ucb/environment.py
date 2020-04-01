from prova_ucb.conversion_rate import Logistic, Product1Season1
import numpy as np

class EnvironmentUCB(object):

    def __init__(self, arms, variance=0):
        self.arms = arms
        self.n_arms = self.arms.shape[0]
        self.curve = Product1Season1()
        self.variance = variance

    def round(self, pulled_arm):
        # print(pulled_arm, self.curve.get_probability(pulled_arm))
        cr = self.curve.get_probability(pulled_arm)
        if self.variance:
            return np.random.normal(cr, self.variance) * pulled_arm
        else:
            return cr * pulled_arm

    def opt_reward(self):
        tmp = []
        for arm in self.arms:
            tmp.append(self.curve.get_probability(arm) * arm)
        max_reward = np.amax(tmp)
        return max_reward





