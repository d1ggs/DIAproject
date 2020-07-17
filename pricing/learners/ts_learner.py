from pricing.learners.learner import Learner
import numpy as np


class TSLearner(Learner):
    """Implements a Thompson Sampling learner"""

    def __init__(self, prices: list):
        """
        :param prices: list of possible prices
        """
        self.n_arms = len(prices)
        super().__init__(prices)
        self.prices = prices
        self.beta_parameters = np.ones((self.n_arms, 2))  # initialization of the beta parameters

    def pull_arm(self):
        # Explore all arms once
        if np.min(self.pull_count) == 0:
            idx = np.argmin(self.pull_count)
            return idx

        # If the exploration has been done decide based on the samples weighed by the corresponding price
        samples = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        vec = []
        for i in range(len(samples)):
            vec.append(samples[i] * self.prices[i])
        idx = np.argmax(np.array(vec))
        return idx

    def update(self, pulled_arm, reward):
        """
        :param pulled_arm: index of the pulled arm
        :param reward: reward of the pulled arm
        """
        self.t += 1

        # Update the observations list
        self.update_observations(pulled_arm, reward)

        # Update the beta distributions for the pulled arm
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1 - reward

    def get_last_best_price(self):
        """
        The method computes the price which in expectation provides the largest value if proposed to all users
        :return: the value of the best price
        """

        if self.t == 0:
            return np.random.choice(self.prices)

        expected_reward = np.copy(self.mean_reward_per_arm)
        for i in range(len(self.mean_reward_per_arm)):
            expected_reward[i] *= self.prices[i]

        best_price_index = np.argmax(expected_reward).squeeze()

        beta = self.beta_parameters[best_price_index]
        best_arm_conversion_prob = beta[0] / (beta[0] + beta[1])

        return self.prices[best_price_index] * best_arm_conversion_prob
