from pricing.learners.ts_learner import TSLearner
import numpy as np


class SWTSLearner(TSLearner):
    """Implements a Sliding-Window Thompson Sampling learner"""

    def __init__(self, prices: list, horizon: int, const=1.0, verbose=False):
        """
        :param n_arms: the number of arms in the setting
        :param horizon: the number of samples that are to be considered when updating distributions
        :param const: adjustment constant to consider the fact that at each time step we pull more than once
        """
        super().__init__(prices)
        self.window_size = 4 * int(np.sqrt(horizon * const))
        self.pulled_arms = np.array([])
        if verbose:
            print("SW-TS using window size:", self.window_size)

    def update(self, pulled_arm: int, reward: float):
        """
        :param pulled_arm: the index of the am pulled at the current time step
        :param reward: the collected reward for the pulled arm
        """
        self.t += 1  # Increase time step
        self.update_observations(pulled_arm, reward)  # Store new observations and update distributions
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)  # Store history of pulled arms

        window = min(self.t, self.window_size)

        # Count how many samples we have collected for the arm that are actually in the window,
        # compute the coumulative reward for each arm and update the distributions.
        # This makes the algorithm discard old samples
        n_samples = np.sum(self.pulled_arms[-window:] == pulled_arm)
        if n_samples != 0:
            cum_rew = np.sum(self.rewards_per_arm[pulled_arm][-n_samples:])
        else:
            cum_rew = 0

        self.beta_parameters[pulled_arm, 0] = max(1, cum_rew + reward)
        self.beta_parameters[pulled_arm, 1] = max(1, (n_samples - cum_rew) + (1 - reward))

        # else:
        #     self.beta_parameters[arm, 0] = cum_rew
        #     self.beta_parameters[arm, 1] = n_samples - cum_rew
