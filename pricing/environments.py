import numpy as np
import math
import matplotlib.pyplot as plt
from pricing.conversion_rate import ProductConversionRate
from abc import ABC


class Environment(ABC):
    def round(self, pulled_arm: int):
        pass

    def opt_reward(self):
        pass

    def get_inst_regret(self, arm):
        pass

    def plot(self):
        pass


class StationaryEnvironment(Environment):
    """
    Models a stationary MAB setting, storing probabilities of success for each arm and providing
    stochastic rewards for each arm
    """

    def __init__(self, prices: list, curve: ProductConversionRate):
        """
        :param prices: the number of arms of the MAB setting
        :param curve: object responsible for computing the conversion rate probabilities
        """

        super().__init__()
        self.n_arms = len(prices)
        self.arms = np.arange(self.n_arms)
        self.prices = prices
        self.curve = curve

    def round(self, pulled_arm: int):
        """
        Computes the reward for the pulled arms, drawing from a binomial distribution

        :param: pulled_arm: the index of the arm that has been pulled at this time step
        :return: the sampled reward for the pulled arm

        """
        cr = self.curve.get_probability(pulled_arm)
        return np.random.binomial(1, cr)

    def opt_reward(self):
        tmp = []
        for i in range(self.n_arms):
            tmp.append(self.curve.get_probability(self.arms[i]) * self.prices[i])

        return np.max(tmp), np.argmax(tmp)

    def plot(self):
        """Plot the conversion rate curve"""
        self.curve.plot()

    def get_inst_regret(self, arm):
        return self.opt_reward()[0] - self.prices[arm] * self.curve.get_probability(arm)


class NonStationaryEnvironment(Environment):
    """
    Models a non-stationary MAB setting, storing probabilities of success for each arm and providing
    stochastic rewards for each arm
    """

    def __init__(self, prices: list, curves: list, horizon: int):

        """
        :param prices: the list of prices to be proposed to customers
        :param curves: the list of conversion rate curves, one for each seasonality
        :param horizon: the maximum amount of time steps before the computation stops
        """

        self.n_arms = len(prices)
        self.arms = np.arange(self.n_arms)
        self.curves = curves
        self.prices = prices
        self.t = 0  # Represents the current time step
        self.horizon = horizon

        n_phases = len(self.curves)  # The number of phases is equal to the number of arms
        self.phase_size = math.ceil(self.horizon / n_phases)  # Assuming that all phases have the same size
        self.current_phase = 0
        self.start_new_phase = True

    def round(self, pulled_arm):
        """
        Computes the reward for the pulled arms, drawing from a binomial distribution

        :param: pulled_arm: the index of the arm that has been pulled at this time step
        :return: the sampled reward for the pulled arm
        """
        old_phase = self.current_phase
        cr = self.curves[self.current_phase].get_probability(pulled_arm)

        self.current_phase = math.floor(self.t / self.phase_size)
        if old_phase != self.current_phase:
            self.start_new_phase = True
        z = np.random.binomial(1, cr)
        return z

    def forward_time(self):
        self.t += 1

    def opt_reward(self):
        """
        Returns the optimal price and its index, with respect to the current phase
        """
        tmp = []
        for i in range(self.n_arms):
            tmp.append(self.curves[self.current_phase].get_probability(self.arms[i]) * self.prices[i])
        return np.max(tmp), np.argmax(tmp)

    def get_inst_regret(self, arm):
        """
        Return the regret of the selected arm, computed as the difference between the optimal arm's reward
        and the current arm's reward.
        one.
        :param arm: arm index
        """
        return self.opt_reward()[0] - self.prices[arm] * self.curves[self.current_phase].get_probability(arm)

    def plot(self):
        """
        This method plots the optimal arms trend during the phases.
        """
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

        plt.legend(["phase " + str(i) for i in range(len(self.curves))])
        plt.show()

    def new_phase(self):
        """
        :returns: True is a new phases has started, False otherwise.
        """
        if self.start_new_phase:
            self.start_new_phase = False
            return True
        return False
