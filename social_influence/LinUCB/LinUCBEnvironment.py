import numpy as np

from social_influence.influence_maximisation import GreedyLearner


class LinUCBEnvironment(object):
    """
    Environment for LinUCB learner
    """

    def __init__(self, probability_matrix: np.ndarray):
        """
        :param probability_matrix: edge activation probability matrix representing a social network
        """
        self.probability_matrix = probability_matrix

    def round(self, pulled_arm: list):
        """
        :return: 1 if the pulled edge has been activated
        :param pulled_arm: index of the selected arm
        """
        p = np.random.random()
        if self.probability_matrix[pulled_arm[0], pulled_arm[1]] > p:
            return 1
        else:
            return 0

    def opt(self, budget, mc_simulations, n_steps, parallel=False):
        """
        :return: the optimal seed, calculated using a Greedy Learner over the probability matrix

        :param budget: budget of the social network
        :param mc_simulations: number of Monte-Carlo simulations
        :param n_steps: maximum number of steps in a simulation
        :param parallel: True if the algorithm is executed with parallel threading
        """
        greedy_learner = GreedyLearner(self.probability_matrix, self.probability_matrix.shape[0])
        if parallel:
            seed, best_reward = greedy_learner.parallel_fit(budget, mc_simulations, n_steps)
        else:
            seed, best_reward = greedy_learner.fit(budget, mc_simulations, n_steps)

        return best_reward - budget, seed
