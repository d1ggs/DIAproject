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
        self.social1_learner = GreedyLearner(social1.get_matrix(), social1.get_n_nodes())
        self.social2_learner = GreedyLearner(social2.get_matrix(), social2.get_n_nodes())
        self.social3_learner = GreedyLearner(social3.get_matrix(), social3.get_n_nodes())
        assert(budget_total >= 3)
        self.budget_total = budget_total
        self.mc_simulations = mc_simulations
        self.n_steps_montecarlo = n_steps_montecarlo

        # Pre-compute social influence results and then transform it into a dictionary budget->influence
        influence_results1, influence_results2, influence_results3 = self.step_joint_influence()
        cumulative_influence1= self.dictionary_creation(influence_results1)
        cumulative_influence2 = self.dictionary_creation(influence_results2)
        cumulative_influence3 = self.dictionary_creation(influence_results3)
        self.cumulative_influence = [cumulative_influence1, cumulative_influence2, cumulative_influence3]
        if self.verbose:
            print(self.cumulative_influence[1], self.cumulative_influence[2], self.cumulative_influence[3])

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

    def joint_influence(self):
        """"
        Pre-computes social influence at each step for each social
        """
        if self.verbose:
            print("Pre-computing social influence")
        # results is an array composed by tuple where each tuple is (node_step_i , influence_step_i)
        results1 = self.social1_learner.cumulative_parallel_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        results2 = self.social2_learner.cumulative_parallel_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        results3 = self.social3_learner.cumulative_parallel_fit(self.budget_total-2, self.mc_simulations, self.n_steps_montecarlo)
        if self.verbose:
            print(results1, results2, results3)
        return results1, results2, results3

    def step_joint_influence(self):
        """"
        Pre-computes social influence at each step for each social
        """
        if self.verbose:
            print("Pre-computing social influence")
        # results is an array composed by tuple where each tuple is (node_step_i , influence_step_i)
        results1 = []
        best_node1, influence_1 = self.social1_learner.step_fit( self.mc_simulations, self.n_steps_montecarlo)
        results1.append((best_node1, influence_1))
        best_node2, influence_2 = self.social1_learner.step_fit( self.mc_simulations, self.n_steps_montecarlo, resume_seeds=[best_node1])
        results1.append((best_node2, influence_2))


        results2 = []
        best_node1, influence_1 = self.social2_learner.step_fit( self.mc_simulations, self.n_steps_montecarlo)
        results2.append((best_node1, influence_1))
        best_node2, influence_2 = self.social2_learner.step_fit( self.mc_simulations, self.n_steps_montecarlo, resume_seeds=[best_node1])
        results2.append((best_node2, influence_2))

        results3 = []
        best_node1, influence_1 = self.social3_learner.step_fit( self.mc_simulations, self.n_steps_montecarlo)
        results3.append((best_node1, influence_1))
        best_node2, influence_2 = self.social3_learner.step_fit( self.mc_simulations, self.n_steps_montecarlo, resume_seeds=[best_node1])
        results3.append((best_node2, influence_2))
        if self.verbose:
            print(results1, results2, results3)
        return results1, results2, results3

    def joint_influence_maximization(self, weights=None):
        """"

        @return: array [budget1, budget2, budget3] with maximized joint social influence value respecting the constraint
        """
        # Instantiate a np array of ones (1,1,1), to impose 1 to be the minimum budget of a social network
        budget = [1, 1, 1]

        if weights is None:
            weights = [1, 1, 1]

        # While the sum of a budget is not equal to the maximum budget, continues the incrementation. Then returns the optimal value
        while not sum(budget) == self.budget_total:
            # Increments the budget of the social with the maximum increase between the three
            argument_to_increment = np.argmax([(self.cumulative_influence[0][budget[0]+1][1] - self.cumulative_influence[0][budget[0]][1]) * weights[0],
                                               (self.cumulative_influence[1][budget[1]+1][1] - self.cumulative_influence[1][budget[1]][1]) * weights[1],
                                               (self.cumulative_influence[2][budget[2]+1][1] - self.cumulative_influence[2][budget[2]][1]) * weights[2]])
            budget[argument_to_increment] += 1

            if sum(budget) != self.budget_total:

                if argument_to_increment == 0:
                    learner = self.social1_learner
                    cumulative_influence = self.cumulative_influence[1]
                elif argument_to_increment == 1:
                    learner = self.social2_learner
                    cumulative_influence = self.cumulative_influence[2]
                elif argument_to_increment == 2:
                    learner = self.social3_learner
                    cumulative_influence = self.cumulative_influence[3]

                
                resume_seeds = []
                for i in range(1,int(budget[argument_to_increment])+1):
                    resume_seeds.append(cumulative_influence[i][0])

                best_node, influence = learner.step_fit(self.mc_simulations, self.n_steps_montecarlo, resume_seeds=resume_seeds)
                self.cumulative_influence[argument_to_increment][budget[argument_to_increment]+1] = [best_node, influence]

            
        joint_influence =  self.cumulative_influence[0][budget[0]][1] + self.cumulative_influence[1][budget[1]][1] + self.cumulative_influence[2][budget[2]][1]
        
        if self.verbose:
            print("Optimal budget: ", budget, "Optimal joint influence: ", joint_influence)

        seeds1 = np.array([v[0] for k,v in self.cumulative_influence[0].items()])[:budget[0]]
        seeds2 = np.array([v[0] for k,v in self.cumulative_influence[1].items()])[:budget[1]]
        seeds3 = np.array([v[0] for k,v in self.cumulative_influence[2].items()])[:budget[2]]
        seeds = {0: seeds1,
                 1: seeds2,
                 2: seeds3}

        print(seeds)

        return budget, joint_influence, seeds












