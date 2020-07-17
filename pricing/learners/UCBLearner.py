import numpy as np
from pricing.learners.learner import Learner


class UCBLearner(Learner):
    """Implements a UCB learner"""

    def __init__(self, prices: list, constant=1):
        """
        :param prices: list of possible prices
        :param constant: UCB constant
        """
        n_arms = len(prices)
        super().__init__(prices)
        self.const = constant
        self.prices = prices
        self.upper_bounds = np.ones(n_arms) * np.inf  # initialization of the upper bounds

    def pull_arm(self):
        """
        Returns the arm with largest upper bound
        """
        idx = np.argmax(self.upper_bounds)
        return idx

    def update(self, pulled_arm, reward):
        """
        :param pulled_arm: index of the pulled arm
        :param reward: reward of the pulled arm
        """

        # update the observation lists
        self.update_observations(pulled_arm, reward * self.prices[pulled_arm])
        self.t += 1
        x = self.mean_reward_per_arm[pulled_arm]
        sqrt_term = np.sqrt(2 * np.log(float(self.t)) / self.pull_count[pulled_arm])

        # compute the new upper bound
        self.upper_bounds[pulled_arm] = x + self.const * sqrt_term * self.prices[pulled_arm]


class SWUCBLearner(UCBLearner):
    """Class implementing SW-UCB# state-of-the-art bandit for non-stationary contexts"""

    def __init__(self, horizon: int, prices: list, const=1, alpha=2, verbose=False):
        """
        :param horizon: the number of samples that are to be considered when updating distributions
        :param prices: the amount of money gained pulling each arm if the user buys
        :param const: adjustment constant to consider the fact that at each time step we pull more than once
        :param alpha: SW-UCB alpha parameter
        """
        super().__init__(prices)
        self.horizon = horizon * const
        self.alpha = alpha
        self.pulls = []
        self.tau = 4 * int(np.sqrt(horizon * const))  # compute the window size
        if verbose:
            print("SW-UCB using window size:", self.tau)

    def get_times_pulled(self, arm):
        """
        Compute how many times an arm has been pulled inside the sliding window
        """
        start = max(0, self.t - self.tau + 1)
        return self.pulls[start:].count(arm)

    def pull_arm(self):

        # pull each arm at least once
        for i in range(self.n_arms):
            if len(self.rewards_per_arm[i]) == 0:
                return i

        # pull the arm with largest upper bound
        idx = np.argmax(self.upper_bounds)
        return idx

    def update(self, pulled_arm, reward: float):
        """
        Update the confidence bounds over the arms
        :param pulled_arm: index of the pulled arm
        :param reward: reward of the pulled arm
        """
        self.pulls.append(pulled_arm)

        # updates the list of past rewards and pulls by removing the older ones
        if self.t > self.tau:
            removed_arm = self.pulls[0]
            self.pulls = self.pulls[1:]
            if self.pull_count[removed_arm] > 0:
                self.pull_count[removed_arm] -= 1
            self.rewards_per_arm[removed_arm] = self.rewards_per_arm[removed_arm][1:]

        # update the observation lists
        self.update_observations(pulled_arm, reward * self.prices[pulled_arm])

        # get the past rewards for the selected arm
        past_rewards = self.rewards_per_arm[pulled_arm]

        # get the mean value of the rewards
        r = self.mean_reward_per_arm[pulled_arm]

        # Compute the uncertainty bound for the window
        log_argmument = min(self.t, self.tau) if self.t > 0 else 1
        sqrt_term = np.sqrt(self.alpha * np.log(log_argmument) / len(past_rewards))

        # Update the overall bound
        self.upper_bounds[pulled_arm] = r + self.const * sqrt_term * self.prices[pulled_arm]
        self.t += 1
