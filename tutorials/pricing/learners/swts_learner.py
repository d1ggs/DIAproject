from tutorials.pricing.learners.ts_learner import TSLearner
import numpy as np


class SWTSLearner(TSLearner):
    """Implements a Sliding-Window Thompson Sampling learner"""

    def __init__(self, n_arms: int, window_size: int):
        """
        :param n_arms: the number of arms in the setting
        :param window_size: the number of samples that are to be considered when updating distributions
        """
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])

    def update(self, pulled_arm: int, reward: float):
        """
        :param pulled_arm: the index of the am pulled at the current time step
        :param reward: the collected reward for the pulled arm
        """
        self.t += 1  # Increase time step
        self.update_observations(pulled_arm, reward)  # Store new observations and update distributions
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)  # Store history of pulled arms

        # Count how many samples we have collected for the arm that are actually in the window,
        # compute the coumulative reward for each arm and update the distributions.
        # This makes the algorithm discard old samples
        for arm in range(0, self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            if n_samples != 0:
                cum_rew = np.sum(self.rewards_per_arm[arm][-n_samples:])
            else:
                cum_rew = 0

            self.beta_parameters[arm, 0] = cum_rew + 1.0
            self.beta_parameters[arm, 1] = n_samples - cum_rew + 1.0