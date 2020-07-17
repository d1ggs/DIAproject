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
        self.window_size = 4 * int(np.sqrt(horizon * const)) # compute the window size
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


    def get_last_best_price(self):
        """
        The method computes the price which in expectation provides the largest value if proposed to all users
        :return: the value of the best price
        """

        if self.t == 0:
            return np.random.choice(self.prices)

        expected_rewards = []
        past_rewards = np.copy(self.rewards_per_arm)
        start = max(0, self.t - self.window_size + 1)
        for i in range(self.n_arms):
            past_rewards[i] = past_rewards[i][start:]
            if len(past_rewards[i])>0:
                r = np.mean(past_rewards[i])*self.prices[i]
            else: r=0
            expected_rewards.append(r)


        best_price_index = np.argmax(expected_rewards).squeeze()

        beta = self.beta_parameters[best_price_index]
        best_arm_conversion_prob = beta[0] / (beta[0] + beta[1])

        return self.prices[best_price_index] * best_arm_conversion_prob