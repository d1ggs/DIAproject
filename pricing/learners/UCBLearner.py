import numpy as np
from pricing.learners.learner import Learner


class UCBLearner(Learner):
    def __init__(self, n_arms: int, prices: list, constant=1):
        super().__init__(n_arms)

        assert n_arms == len(prices), "Number of prices different from the number of arms"

        self.const = constant
        self.prices = prices
        self.upper_bounds = np.ones(n_arms) * np.inf

    def pull_arm(self):
        idx = np.argmax(self.upper_bounds)
        return idx

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.t += 1
        x = np.mean(self.rewards_per_arm[pulled_arm])
        self.upper_bounds[pulled_arm] = x + self.const * np.sqrt((2 * np.log(self.t)) / len(self.rewards_per_arm[pulled_arm]))


class SWUCBLearner(UCBLearner):
    """Class implementing SW-UCB# state-of-the-art bandit for non-stationary contexts"""
    def __init__(self, n_arms: int, horizon: int, prices: list, const: int, alpha=2):
        """
        :param n_arms: number of conversion rates to be learned
        :param horizon: the number of samples that are to be considered when updating distributions
        :param prices: the amount of money gained pulling each arm if the user buys
        :param alpha: SW-UCB alpha parameter
        :param const: adjustment constant to consider the fact that at each time step we pull more than once
        """
        super().__init__(n_arms, prices)
        self.horizon = horizon * const
        self.alpha = alpha
        self.pulls = []
        self.tau = int(4 * np.sqrt(self.horizon * np.log(self.horizon)))
        print("SW-UCB using window size:", self.tau)

    def get_times_pulled(self, arm):
        """Compute how many times an arm has been pulled inside the sliding window"""
        start = max(0, self.t - self.tau + 1)
        return self.pulls[start:].count(arm)

    def update_observations(self, pulled_arm: int, reward: float):
        # Store the reward, putting 0s to those arms that were not pulled
        for i in range(len(self.rewards_per_arm)):
            self.rewards_per_arm[i].append(reward if i == pulled_arm else 0)

        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update(self, pulled_arm, reward):
        """Update the confidence bounds over the arms"""
        self.pulls.append(pulled_arm)
        self.update_observations(pulled_arm, reward * self.prices[pulled_arm])
        self.t += 1

        # Compute the empirical mean only for those pulls that are in the window
        start = max(0, self.t - self.tau + 1)
        r = np.sum(self.rewards_per_arm[pulled_arm][start:]) / self.get_times_pulled(pulled_arm)

        # Compute the uncertainty bound for the window
        c = np.sqrt(self.alpha * np.log(min(self.t, self.tau))/self.get_times_pulled(pulled_arm))

        # Update the overall bound
        self.upper_bounds[pulled_arm] = r + self.const * c * self.prices[pulled_arm]
