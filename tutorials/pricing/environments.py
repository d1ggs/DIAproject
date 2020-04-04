import numpy as np

from tutorials.pricing.conversion_rate import Product1Season1


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

    def __init__(self, arms: list, prices: list, variance=0):
        self.arms = arms
        self.prices = prices
        self.n_arms = len(self.arms)
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
        self.phase_size = self.horizon / n_phases  # Assuming that all phases have the same size

    def round(self, pulled_arm):
        """
        Computes the reward for the pulled arms, drawing from a binomial distribution

        :param: pulled_arm: the index of the arm that has been pulled at this time step
        :return: the sampled reward for the pulled arm
        """
        current_phase = int(self.t / self.phase_size)

        cr = self.curves[current_phase].get_probability(pulled_arm)
        self.t += 1
        return np.random.binomial(1, cr)

    def opt_reward(self):
        current_phase = int(self.t / self.phase_size)

        tmp = []
        for i in range(self.n_arms):
            tmp.append(self.curves[current_phase].get_probability(self.arms[i]) * self.prices[i])

        return self.prices[np.argmax(tmp)]
