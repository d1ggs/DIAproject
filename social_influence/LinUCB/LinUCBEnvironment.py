import numpy as np

from social_influence.influence_maximisation import GreedyLearner


class LinUCBEnvironment():
    def __init__(self, probability_matrix):
        self.probability_matrix = probability_matrix

    def round(self, pulled_arm):
        p = np.random.random()
        if self.probability_matrix[pulled_arm[0], pulled_arm[1]] > p:
            return 1
        else:
            return 0


    def opt(self, budget, mc_simulations, n_steps, parallel=False):
        """

        :return: optimal seed
        """
        greedy_learner = GreedyLearner(self.probability_matrix, self.probability_matrix.shape[0])
        if parallel:
            seed, best_reward = greedy_learner.parallel_fit(budget, mc_simulations, n_steps)
        else:
            seed, best_reward = greedy_learner.fit(budget, mc_simulations, n_steps)

        return best_reward - budget, seed
