import numpy as np
from tutorials.pricing.learners.learner import Learner


class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.upper_bounds = np.ones(n_arms) * np.inf

    def pull_arm(self):
        idx = np.argmax(self.upper_bounds)
        return idx

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.t += 1
        x = np.mean(self.rewards_per_arm[pulled_arm])
        self.upper_bounds[pulled_arm] = x + np.sqrt((2 * np.log(self.t)) / len(self.rewards_per_arm[pulled_arm]))


class SWUCBLearner(UCBLearner):
    """Class implementing SW-UCB# state-of-the-art bandit for non-stationary contexts"""
    def __init__(self, n_arms: int, lamda: float, alpha: float):
        assert lamda >= 0, 'lambda must be equal or greater than 0'
        assert 1 >= alpha > 0, 'alpha must satisfy 0 < alpha <= 1'
        super().__init__(n_arms)
        self.lamda = lamda
        self.alpha = alpha
        self.pulls = []

    def get_tau(self):
        """Compute the size of the sliding window"""
        return min(self.t, np.ceil(self.lamda * (self.t ** self.alpha)))

    def get_times_pulled(self, arm):
        """Compute how many times an arm has been pulled inside the sliding window"""
        start = self.t - self.get_tau() + 1
        return self.pulls[start:].count(arm)

    def update_observations(self, pulled_arm: int, reward: float):
        # Store the reward, putting 0s to those arms that were not pulled
        for i in range(len(self.rewards_per_arm)):
            self.rewards_per_arm[i].append(reward if i == pulled_arm else 0)

        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update(self, pulled_arm, reward):
        """Update the confidence bounds over the arms"""
        self.pulls.append(pulled_arm)
        self.update_observations(pulled_arm, reward)
        self.t += 1

        # Compute the empirical mean only for those pulls that are in the window

        start = self.t - self.get_tau() + 1
        r = 1/self.get_times_pulled(pulled_arm) * np.sum(self.rewards_per_arm[pulled_arm][start:])

        # Compute the uncertainty bound for the window
        c = np.sqrt(((1+self.alpha) * np.log(self.t))/self.get_times_pulled(pulled_arm))

        # Update the overall bound
        self.upper_bounds[pulled_arm] = r + c
