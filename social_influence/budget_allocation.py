import numpy as np
from social_influence.influence_maximisation import GreedyLearner

class GreedyBudgetAllocation(object):

    def __init__(self, social1, social2, social3, budget_total, mc_simulations, n_steps_montecarlo, verbose=False):

        """
        In the main we need to pass 3 social media objects through the main
        Parameters
        ----------
        social1
        social2
        social3: three social network object to be passed
        budget_total
        """

        self.verbose = verbose
        if self.verbose:
            print(social1.get_matrix().shape, social1.get_n_nodes())
        self.social1_learner = GreedyLearner(social1.get_matrix(), social1.get_n_nodes())
        self.social2_learner = GreedyLearner(social2.get_matrix(), social2.get_n_nodes())
        self.social3_learner = GreedyLearner(social3.get_matrix(), social3.get_n_nodes())
        assert(budget_total >= 3)
        self.budget_total = budget_total
        self.mc_simulations = mc_simulations
        self.n_steps_montecarlo = n_steps_montecarlo

        # Pre-compute social influence results and then transform it into a dictionary budget->influence
        influence_results1, influence_results2, influence_results3 = self.joint_influence()
        cumulative_influence1, seed1 = self.dictionary_creation(influence_results1)
        cumulative_influence2, seed2 = self.dictionary_creation(influence_results2)
        cumulative_influence3, seed3 = self.dictionary_creation(influence_results3)
        self.cumulative_influence = [cumulative_influence1, cumulative_influence2, cumulative_influence3]
        self.max_seeds = [seed1, seed2, seed3]
        if self.verbose:
            print(self.cumulative_influence[0], self.cumulative_influence[1], self.cumulative_influence[2])


    @staticmethod
    def dictionary_creation(influence_tuples):
        """
        Discards seed informations and converts it into a dictionary
        @param influence_tuples: array of tuples (seed, influence)
        @return: dictionary {budget: influence}
        """
        dictionary = {}
        budget = 1
        seeds = []
        for seed, influence in influence_tuples:
            dictionary[budget] = influence
            budget += 1
            seeds.append(seed)
        return dictionary, seeds

    def joint_influence(self):
        """"
        Pre-computes social influence at each step for each social
        """
        if self.verbose:
            print("Pre-computing social influence")
        # results is an array composed by tuple where each tuple is (node_step_i , influence_step_i)
        results1 = self.social1_learner.cumulative_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        results2 = self.social2_learner.cumulative_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        results3 = self.social3_learner.cumulative_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        if self.verbose:
            print(results1, results2, results3)
        return results1, results2, results3

    def joint_influence_maximization(self, weights=None, verbose=False):
        """"

        @return: array [budget1, budget2, budget3] with maximized joint social influence value respecting the constraint
        """
        if weights is None:
            weights = [1, 1, 1]

        # Instantiate a np array of ones (1,1,1), to impose 1 to be the minimum budget of a social network

        budget = [1, 1, 1]

        # While the sum of a budget is not equal to the maximum budget, continues the incrementation. Then returns the optimal value
        while not sum(budget) == self.budget_total:
            # Increments the budget of the social with the maximum increase between the three
            argument_to_increment = np.argmax([(self.cumulative_influence[0][budget[0]+1] - self.cumulative_influence[0][budget[0]]) * weights[0],
                                               (self.cumulative_influence[1][budget[1]+1] - self.cumulative_influence[1][budget[1]]) * weights[1],
                                               (self.cumulative_influence[2][budget[2]+1] - self.cumulative_influence[2][budget[2]]) * weights[2]])

            budget[argument_to_increment] += 1

        if verbose:
            print("Optimal budget: ", budget)
            print("Optimal joint influence: ", self.cumulative_influence[0][budget[0]] + self.cumulative_influence[1][budget[1]] + self.cumulative_influence[2][budget[2]])

        seeds = {0: np.array(self.max_seeds[0][:budget[0]]),
                 1: np.array(self.max_seeds[1][:budget[1]]),
                 2: np.array(self.max_seeds[2][:budget[2]])}

        return budget, seeds











