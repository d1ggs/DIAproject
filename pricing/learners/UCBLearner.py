import numpy as np
from pricing.learners.learner import Learner


class UCBLearner(Learner):
    def __init__(self, prices: list, constant=1):
        n_arms = len(prices)
        super().__init__(prices)
        self.const = constant
        self.prices = prices
        self.upper_bounds = np.ones(n_arms) * np.inf

    def pull_arm(self):
        idx = np.argmax(self.upper_bounds)
        return idx

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward*self.prices[pulled_arm])
        self.t += 1
        x = self.mean_reward_per_arm[pulled_arm]
        sqrt_term = np.sqrt(2 * np.log(float(self.t)) / self.pull_count[pulled_arm])
        # print(float(n_times_pulled))
        # print(self.t)
        # print(sqrt_term)
        self.upper_bounds[pulled_arm] = x + self.const * sqrt_term * self.prices[pulled_arm]
        # print(self.upper_bounds)
       # return self



class SWUCBLearner(UCBLearner):
    """Class implementing SW-UCB# state-of-the-art bandit for non-stationary contexts"""
    def __init__(self, horizon: int, prices: list, const=1, alpha=2, verbose=False):
        """
        :param n_arms: number of conversion rates to be learned
        :param horizon: the number of samples that are to be considered when updating distributions
        :param prices: the amount of money gained pulling each arm if the user buys
        :param alpha: SW-UCB alpha parameter
        :param const: adjustment constant to consider the fact that at each time step we pull more than once
        """
        super().__init__(prices)
        self.horizon = horizon * const
        self.alpha = alpha
        self.pulls = []
        #self.tau = int(4 * np.sqrt(self.horizon * np.log(self.horizon)))
        self.tau=4 * int(np.sqrt(horizon*const))
        if verbose:
            print("SW-UCB using window size:", self.tau)

    def get_times_pulled(self, arm):
        """Compute how many times an arm has been pulled inside the sliding window"""
        start = max(0, self.t - self.tau + 1)
        return self.pulls[start:].count(arm)

    # def update_observations(self, pulled_arm: int, reward: float):
    #     # Store the reward, putting 0s to those arms that were not pulled
    #     for i in range(len(self.rewards_per_arm)):
    #         self.rewards_per_arm[i].append(reward if i == pulled_arm else 0)
    #
    #     self.collected_rewards = np.append(self.collected_rewards, reward)

    def pull_arm(self):
        #start = max(0, self.t - self.tau )
        for i in range(self.n_arms):
            if len(self.rewards_per_arm[i])==0:
                return i

        idx = np.argmax(self.upper_bounds)

        return idx


    def update(self, pulled_arm, reward:float):
        """Update the confidence bounds over the arms"""
        self.pulls.append(pulled_arm)

        if self.t > self.tau:
            removed_arm = self.pulls[0]
            self.pulls=self.pulls[1:]
            if  self.pull_count[removed_arm]>0:
                self.pull_count[removed_arm]-=1
            self.rewards_per_arm[removed_arm] = self.rewards_per_arm[removed_arm][1:]

        self.update_observations(pulled_arm, reward*self.prices[pulled_arm])


        # Compute the empirical mean only for those pulls that are in the window
        past_rewards = self.rewards_per_arm[pulled_arm]

        #start = max(0, self.t - self.tau)
        #past_rewards = past_rewards[start:]
        r = self.mean_reward_per_arm[pulled_arm]

        # Compute the uncertainty bound for the window
        log_argmument = min(self.t,self.tau) if self.t>0 else 1
        sqrt_term = np.sqrt(self.alpha * np.log(log_argmument)/len(past_rewards))


        # Update the overall bound
        self.upper_bounds[pulled_arm] = r + self.const * sqrt_term * self.prices[pulled_arm]
        self.t += 1