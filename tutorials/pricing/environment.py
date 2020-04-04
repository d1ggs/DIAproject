from tutorials.pricing.conversion_rate import Product1Season1
import numpy as np

class EnvironmentUCB(object):

    def __init__(self, arms, variance=0):
        self.arms = arms
        self.prices = [63, 76, 10, 8, 53, 21]
        self.n_arms = self.arms.shape[0]
        self.curve = Product1Season1()
        self.variance = variance

    def round(self, pulled_arm):
        # print(pulled_arm, self.curve.get_probability(pulled_arm))
        cr = self.curve.get_probability(pulled_arm)
        return np.random.binomial(1, cr)

    def opt_reward(self):
        tmp = []
        for i in range(len(self.arms)):
            tmp.append(self.curve.get_probability(self.arms[i]) * self.prices[i])
        max_reward = np.amax(tmp)
        return max_reward





