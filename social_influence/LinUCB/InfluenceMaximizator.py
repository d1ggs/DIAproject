
import numpy as np
import numpy as np
from social_influence.influence_maximisation import GreedyLearner

class InfluenceMaximizator():
    def __init__(self, feature_matrix,n_experiment,budget, mc_simulations, n_steps):
        self.feature_matrix = feature_matrix
        self.theta = np.atleast_2d(np.zeros(self.feature_matrix.shape[2]))
        self.n_experiment = n_experiment
        self.prob_matrix = np.zeros((self.feature_matrix.shape[0],self.feature_matrix.shape[0]))
        self.budget = budget
        self.mc_simulations = mc_simulations
        self.n_steps = n_steps

    def update_tetha(self,extimated_tetha):
        self.theta += extimated_tetha.T

    def __calc_prob_matrix(self):
        self.theta = self.theta/self.n_experiment
        for i in range(self.feature_matrix.shape[0]):
            for j in range(self.feature_matrix.shape[0]):
                self.prob_matrix[i,j] = np.dot(self.feature_matrix[i,j,:],self.theta.T)



    def find_best_seeds(self, parallel = True):
        self.__calc_prob_matrix()
        greedy_learner = GreedyLearner(self.prob_matrix, self.feature_matrix.shape[0])
        if parallel:
            seed, reward = greedy_learner.parallel_fit(self.budget,self.mc_simulations,self.n_steps)
        else:
            seed, reward = greedy_learner.fit(self.budget, self.mc_simulations, self.n_steps)
        return seed, reward

