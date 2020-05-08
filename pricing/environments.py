import numpy as np
import math
from pricing.conversion_rate import Product1Season1
import matplotlib.pyplot as plt


class StationaryEnvironment(object):
    """
    Models a stationary MAB setting, storing probabilities of success for each arm and providing
    stochastic rewards for each arm
    """

    def __init__(self, n_arms: int, probabilities: np.ndarray):
        """
        :param n_arms: the number of arms of the MAB setting
        :param probabilities: the success probability for each Bernoulli distribution associated to an arm
        """

        super().__init__()
        self.n_arms = n_arms
        self.probabilities = probabilities


    def round(self, pulled_arm: int):
        """
        Computes the reward for the pulled arms, drawing from a binomial distribution

        :param: pulled_arm: the index of the arm that has been pulled at this time step
        :return: the sampled reward for the pulled arm

        """
        reward = np.random.binomial(1, self.probabilities[
            pulled_arm])  # The first parameter is 1 because it is a Bernoulli
        return reward





class EnvironmentUCB(object):

    def __init__(self, arms: list, prices: list):
        self.arms = arms
        self.prices = prices
        self.n_arms = len(self.arms)
        self.curve = Product1Season1()
        tmp = []
        for i in range(len(self.arms)):
            tmp.append(self.curve.get_probability(self.arms[i]) * self.prices[i])
        self.opt_reward = np.max(tmp)


    def round(self, pulled_arm):
        # print(pulled_arm, self.curve.get_probability(pulled_arm))
        cr = self.curve.get_probability(pulled_arm)
        return np.random.binomial(1, cr)

    def get_inst_regret(self, arm):
        return self.opt_reward - self.prices[arm] * self.curve.get_probability(arm)


class NonStationaryEnvironment(object):
    """
    Models a non-stationary MAB setting, storing probabilities of success for each arm and providing
    stochastic rewards for each arm
    """

    def __init__(self, arms: list, prices: list, curves: list, horizon: int):
        """
        :param n_arms, the number of arms of the MAB setting
        :param probabilities, the success probability for each Bernoulli distribution associated to an arm
        :param horizon: the maximum amount of time steps before the computation stops
        """

        self.n_arms = len(arms)
        self.arms = arms
        self.curves = curves
        self.prices = prices

        self.t = 0  # Represents the current time step
        self.horizon = horizon

        n_phases = len(self.curves)  # The number of phases is equal to the number of arms
        self.phase_size = math.ceil(self.horizon / n_phases)  # Assuming that all phases have the same size
        self.current_phase=0


    def round(self, pulled_arm):
        """
        Computes the reward for the pulled arms, drawing from a binomial distribution

        :param: pulled_arm: the index of the arm that has been pulled at this time step
        :return: the sampled reward for the pulled arm
        """

        cr = self.curves[self.current_phase].get_probability(pulled_arm)
        self.t += 1
        self.current_phase = math.floor(self.t / self.phase_size)
        return np.random.binomial(1, cr)

    def opt_reward(self):

        tmp = []
        for i in range(self.n_arms):
            tmp.append(self.curves[self.current_phase].get_probability(self.arms[i]) * self.prices[i])

        return np.max(tmp)

    def get_inst_regret(self,arm):
        return self.opt_reward()-self.prices[arm]*self.curves[self.current_phase].get_probability(arm)

    def plot(self):
        plot_expected = []
        for curve in self.curves:
            t = []
            for arm in self.arms:
                t.append(self.prices[arm] * curve.get_probability(arm))
            plot_expected.append(t)

        plt.figure()
        plt.ylabel("Expected return")
        plt.xlabel("arm")
        for plot in plot_expected:
            plt.plot(plot)

        plt.legend(["phase "+str(i) for i in range(len(self.curves))])
        plt.show()

