import numpy as np
from social_influence.influence_maximisation import GreedyLearner

class GreedyBudgetAllocation(object):

    def __init__(self, social1, social2, social3, budget_total,  mc_simulations, n_steps_montecarlo, verbose = False):

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

        print(social1.get_matrix().shape, social1.get_n_nodes())
        social1_learner = GreedyLearner(social1.get_matrix(), social1.get_n_nodes())
        social2_learner = GreedyLearner(social2.get_matrix(), social2.get_n_nodes())
        social3_learner = GreedyLearner(social3.get_matrix(), social3.get_n_nodes())
        self.social_list = [social1_learner, social2_learner, social3_learner]
        assert(budget_total >= 3)
        self.budget_total = budget_total
        self.mc_simulations = mc_simulations
        self.n_steps_montecarlo = n_steps_montecarlo


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
            dictionary[budget] = [seed, influence]
            budget += 1
        return dictionary


    def initialization_step_joint_influence(self):
        """
        Initialization of the algorithm.
        Pre-computes social influence for the first 2 steps for each social and returns a dictionary containing the tuples
        """
        if self.verbose:
            print("Pre-computing social influence")
        # results is an array composed by tuple where each tuple is (node_step_i , influence_step_i)
        results = [[], [], []]
        # Calculates first two steps of influence for each social
        for i in range(3):
            best_node1, influence1 = self.social_list[i].step_fit(self.mc_simulations, self.n_steps_montecarlo)
            results[i].append((best_node1, influence1))
            best_node2, influence2 = self.social_list[i].step_fit(self.mc_simulations, self.n_steps_montecarlo, resume_seeds=[best_node1])
            results[i].append((best_node2, influence2))

        # Returns the dictionary containing the tuples
        results_dict1, results_dict2, results_dict3 = self.dictionary_creation(results[0]),\
                                                      self.dictionary_creation(results[1]), self.dictionary_creation(results[2])
        return results_dict1, results_dict2, results_dict3

    def joint_influence_maximization(self, weights=None):
        """"

        @return: array [budget1, budget2, budget3] with maximized joint social influence value respecting the constraint
        """
                # Pre-compute social influence results and then transform it into a dictionary budget->influence
        cumulative_influence1, cumulative_influence2, cumulative_influence3 = self.initialization_step_joint_influence()
        cumulative_influences = [cumulative_influence1, cumulative_influence2, cumulative_influence3]
        if self.verbose:
            print(cumulative_influences[0], cumulative_influences[1], cumulative_influences[2])
        if weights is None:
            weights = [1, 1, 1]

        # Instantiate a np array of ones (1,1,1), to impose 1 to be the minimum budget of a social network
        budget = [1, 1, 1]

        # While the sum of a budget is not equal to the maximum budget, continues the incrementation. Then returns the optimal value
        while not sum(budget) == self.budget_total:
            # Increments the budget of the 3 social with the maximum increase multiplied by the price weight
            argument_to_increment = np.argmax([(cumulative_influences[0][budget[0] + 1][1] -
                                                cumulative_influences[0][budget[0]][1]) * weights[0],
                                               (cumulative_influences[1][budget[1] + 1][1] -
                                                cumulative_influences[1][budget[1]][1]) * weights[1],
                                               (cumulative_influences[2][budget[2] + 1][1] -
                                                cumulative_influences[2][budget[2]][1]) * weights[2]])
            budget[argument_to_increment] += 1

            # If sum(budget) == self.budget_total the algorithm is done and you don't need to compute the next step
            if sum(budget) != self.budget_total:
                # If not, computes the next step to use in the next iteration of the algorithm
                resume_seeds = []

                # Appends to the resume list the seeds calculated to resume the algorithm quickly
                for i in range(1, int(budget[argument_to_increment]) + 1):
                    resume_seeds.append(cumulative_influences[argument_to_increment][i][0])

                # Computes the next step to use in the next iteration
                best_node, influence = self.social_list[argument_to_increment].step_fit(self.mc_simulations,
                                                                                        self.n_steps_montecarlo,
                                                                                        resume_seeds=resume_seeds)
                cumulative_influences[argument_to_increment][budget[argument_to_increment] + 1] = [best_node, influence]

        # Computes the final joint influence
        joint_influence = cumulative_influences[0][budget[0]][1] + cumulative_influences[1][budget[1]][1] + \
                          cumulative_influences[2][budget[2]][1]

        if self.verbose:
            print("Optimal budget: ", budget, "Optimal joint influence: ", joint_influence)

        # Builds a dictionary with the best seeds allocated with the best budget
        seeds1 = np.array([v[0] for k,v in cumulative_influences[0].items()])[:budget[0]]
        seeds2 = np.array([v[0] for k,v in cumulative_influences[1].items()])[:budget[1]]
        seeds3 = np.array([v[0] for k,v in cumulative_influences[2].items()])[:budget[2]]
        seeds = {0: seeds1,
                 1: seeds2,
                 2: seeds3}

        #print(seeds)

        return budget, joint_influence, seeds


class CumulativeBudgetAllocation(object):

    def __init__(self, matrix1, matrix2, matrix3, budget_total,  mc_simulations, n_steps_montecarlo, verbose = False):

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

        social1_learner = GreedyLearner(matrix1, matrix1.shape[0])
        social2_learner = GreedyLearner(matrix2, matrix2.shape[0])
        social3_learner = GreedyLearner(matrix3, matrix3.shape[0])
        self.social_list = [social1_learner, social2_learner, social3_learner]
        assert(budget_total >= 3)
        self.budget_total = budget_total
        self.mc_simulations = mc_simulations
        self.n_steps_montecarlo = n_steps_montecarlo

        cumulative_influence1, cumulative_influence2, cumulative_influence3 = self.initialization_step_joint_influence()
        self.cumulative_influences = [cumulative_influence1, cumulative_influence2, cumulative_influence3]


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
            dictionary[budget] = [seed, influence]
            budget += 1
        return dictionary


    def initialization_step_joint_influence(self):
        """
        Initialization of the algorithm.
        Pre-computes social influence for each social at maximum budget and returns a dictionary containing the tuples (best_node at step i, influence at step i)
        """
        if self.verbose:
            print("Pre-computing social influence")
        # results is an array composed by tuple where each tuple is (node_step_i , influence_step_i)
        results = [[], [], []]
        # Calculates first two steps of influence for each social
        for i in range(3):
            results[i] = self.social_list[i].cumulative_parallel_fit(self.budget_total-2,self.mc_simulations, self.n_steps_montecarlo)


        # Returns the dictionary containing the tuples
        results_dict1, results_dict2, results_dict3 = self.dictionary_creation(results[0]),\
                                                      self.dictionary_creation(results[1]), self.dictionary_creation(results[2])
        return results_dict1, results_dict2, results_dict3

    def joint_influence_maximization(self, weights=None):
        """"

        @return: array [budget1, budget2, budget3] with maximized joint social influence value respecting the constraint
        """

        if weights is None:
            weights = [1, 1, 1]

        # Instantiate a np array of ones (1,1,1), to impose 1 to be the minimum budget of a social network
        budget = [1, 1, 1]

        # While the sum of a budget is not equal to the maximum budget, continues the incrementation. Then returns the optimal value
        while not sum(budget) == self.budget_total:
            # Increments the budget of the 3 social with the maximum increase multiplied by the price weight
            argument_to_increment = np.argmax([(self.cumulative_influences[0][budget[0] + 1][1] -
                                                self.cumulative_influences[0][budget[0]][1]) * weights[0],
                                               (self.cumulative_influences[1][budget[1] + 1][1] -
                                                self.cumulative_influences[1][budget[1]][1]) * weights[1],
                                               (self.cumulative_influences[2][budget[2] + 1][1] -
                                                self.cumulative_influences[2][budget[2]][1]) * weights[2]])
            budget[argument_to_increment] += 1


        # Computes the final joint influence
        joint_influence = self.cumulative_influences[0][budget[0]][1] + self.cumulative_influences[1][budget[1]][1] + \
                          self.cumulative_influences[2][budget[2]][1]

        if self.verbose:
            print("Optimal budget: ", budget, "Optimal joint influence: ", joint_influence)

        # Builds a dictionary with the best seeds allocated with the best budget
        seeds1 = np.array([v[0] for k,v in self.cumulative_influences[0].items()])[:budget[0]]
        seeds2 = np.array([v[0] for k,v in self.cumulative_influences[1].items()])[:budget[1]]
        seeds3 = np.array([v[0] for k,v in self.cumulative_influences[2].items()])[:budget[2]]
        seeds = {0: seeds1,
                 1: seeds2,
                 2: seeds3}

        #print(seeds)

        return budget, joint_influence, seeds
