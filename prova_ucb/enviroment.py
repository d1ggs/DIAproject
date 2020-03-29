import numpy as np
from conversion_rate import Logistic


class EnviromentUCB():

    def __init__(self, n_arms, n_costumers):
        self.n_arms = n_arms
        self.curve = Logistic(3)
        self.n_costumers = n_costumers

    def round(self, pulled_arm):
        reward = np.random.normal(self.curve.compute(pulled_arm), 0.1)*pulled_arm * self.n_costumers
        return reward
