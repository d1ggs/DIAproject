import numpy as np


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


class NonStationaryEnvironment(StationaryEnvironment):
    """
    Models a non-stationary MAB setting, storing probabilities of success for each arm and providing
    stochastic rewards for each arm
    """
    def __init__(self, n_arms: int, probabilities: np.ndarray, horizon: int):
        """
        :param n_arms, the number of arms of the MAB setting
        :param probabilities, the success probability for each Bernoulli distribution associated to an arm
        :param horizon: the maximum amount of time steps before the computation stops
        """

        # We want to have distinct probabilities for each phase, so we need a probability matrix
        assert len(probabilities.shape) == 2, "The probabilities must be a 2-dimensional array"

        super().__init__(n_arms, probabilities)
        self.t = 0  # Represents the current time step
        self.horizon = horizon

    def round(self, pulled_arm):
        """
        Computes the reward for the pulled arms, drawing from a binomial distribution

        :param: pulled_arm: the index of the arm that has been pulled at this time step
        :return: the sampled reward for the pulled arm
        """
        n_phases = len(self.probabilities)  # The number of phases is equal to the number of arms
        phase_size = self.horizon / n_phases  # Assuming that all phases have the same size
        current_phase = int(self.t / phase_size)

        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1
        return np.random.binomial(1, p)
