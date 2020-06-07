import numpy as np
from social_influence.influence_maximisation import GreedyLearner

class GreedyBudgetAllocation(object):

    def __init__(self, social1, social2, social3, budget_total,  mc_simulations, n_steps_montecarlo, social_prices=None):

        """
        In the main we need to pass 3 social media objects through the main
        Parameters
        ----------
        social1
        social2
        social3: three social network object to be passed
        budget_total
        """

        if social_prices:
            self.social_prices = social_prices
        else:
            self.social_prices = [1, 1, 1]

        print(social1.get_matrix().shape, social1.get_n_nodes())
        self.social1_learner = GreedyLearner(social1.get_matrix(), social1.get_n_nodes())
        self.social2_learner = GreedyLearner(social2.get_matrix(), social2.get_n_nodes())
        self.social3_learner = GreedyLearner(social3.get_matrix(), social3.get_n_nodes())
        assert(budget_total >= 3)
        self.budget_total = budget_total
        self.mc_simulations = mc_simulations
        self.n_steps_montecarlo = n_steps_montecarlo

        # Pre-compute social influence results and then transform it into a dictionary budget->influence
        influence_results1, influence_results2, influence_results3 = self.joint_influence(verbose=True)
        self.cumulative_influence1 = self.dictionary_creation(influence_results1)
        self.cumulative_influence2 = self.dictionary_creation(influence_results2)
        self.cumulative_influence3 = self.dictionary_creation(influence_results3)
        print(self.cumulative_influence1, self.cumulative_influence2, self.cumulative_influence3)


    @staticmethod
    def dictionary_creation(influence_tuples):
        """
        Discards seed informations and converts it into a dictionary
        @param influence_tuples: array of tuples (seed, influence)
        @return: dictionary {budget: influence}
        """
        dictionary = {}
        budget = 1
        for seed, influence in influence_tuples:
            dictionary[budget] = influence
            budget += 1
        return dictionary

    def joint_influence(self, verbose=False):
        """"
        Pre-computes social influence at each step for each social
        """
        if verbose:
            print("Pre-computing social influence")
        # results is an array composed by tuple where each tuple is (node_step_i , influence_step_i)
        results1 = self.social1_learner.cumulative_parallel_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        results2 = self.social2_learner.cumulative_parallel_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        results3 = self.social3_learner.cumulative_parallel_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        if verbose:
            print(results1, results2, results3)
        return results1, results2, results3

    def joint_influence_maximization(self):
        """"

        @return: array [budget1, budget2, budget3] with maximized joint social influence value respecting the constraint
        """
        # Instantiate a np array of ones (1,1,1), to impose 1 to be the minimum budget of a social network
        budget = np.ones(3)

        # While the sum of a budget is not equal to the maximum budget, continues the incrementation. Then returns the optimal value
        while not sum(budget) == self.budget_total:
            # Increments the budget of the social with the maximum increase between the three
            argument_to_increment = np.argmax([(self.cumulative_influence1[budget[0]+1] - self.cumulative_influence1[budget[0]]) * self.social_prices[0],
                                               (self.cumulative_influence2[budget[1]+1] - self.cumulative_influence2[budget[1]]) * self.social_prices[1],
                                               (self.cumulative_influence3[budget[2]+1] - self.cumulative_influence3[budget[2]]) * self.social_prices[2]])
            budget[argument_to_increment] += 1


        print("Optimal budget: ", budget, "Optimal joint influence: ", self.cumulative_influence1[budget[0]] + self.cumulative_influence2[budget[1]] + self.cumulative_influence3[budget[2]])
        return budget










