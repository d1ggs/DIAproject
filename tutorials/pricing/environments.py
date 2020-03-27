import numpy as np


class StationaryEnvironment(object):
    def __init__(self, n_arms: int, probabilities: np.ndarray):
        """
        This class models a MAB setting, storing probabilities of success for each arm and providing stochastic rewards
        for each arm

        :param n_arms, the number of arms of the MAB setting
        :param probabilities, the success probability for each Bernoulli distribution associated to an arm
        """

        super().__init__()
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm: int):
        """Computes the reward for the pulled arms, drawing from a binomial distribution"""
        reward = np.random.binomial(1, self.probabilities[
            pulled_arm])  # the first parameter is 1 because it is a Bernoulli
        return reward


class NonStationaryEnvironment(StationaryEnvironment):
    def __init__(self, n_arms, probalities, horizon):
        super().__init__(n_arms, probalities)
        self.t = 0
        self.horizon = horizon

    def round(self, pulled_arm):
        n_phases = len(self.probabilities)
        phase_size = self.horizon / n_phases
        current_phase = int(self.t / phase_size)

        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1
        return np.random.binomial(1, p)
