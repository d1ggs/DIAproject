from social_influence.influence_maximisation import *
from social_influence.mc_sampling import *



class IMLinUCBLearner():
    def __init__(self, n_features, feature_matrix_edges, budget):
        self.M = np.eye(n_features)
        self.B = np.atleast_2d(np.zeros(n_features)).T
        self.collected_rewards = []
        self.feature_matrix_edges = feature_matrix_edges
        self.sigma = 10
        self.c = 1
        self.budget = budget
        self.n_nodes = feature_matrix_edges.shape[0]


    def project_matrix(self, matrix):
        max = matrix.max()
        min = matrix.min()
        new_max = 1
        new_min = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i,j] = ( (matrix[i,j] - min) / (max - min) ) * (new_max - new_min) + new_min
        return matrix


    def pull_arm(self):
        UCB_matrix = np.zeros((self.n_nodes, self.n_nodes))
        theta = self.sigma ** (-2) * np.dot(np.linalg.inv(self.M), self.B)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                arm = np.atleast_2d(self.feature_matrix_edges[i, j]).T
                ucb = np.dot(theta.T, arm) + self.c * np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
                UCB_matrix[i, j] = ucb[0, 0]
        UCB_matrix = self.project_matrix(UCB_matrix)
        print(UCB_matrix)
        oracolo = GreedyLearner(UCB_matrix, self.n_nodes)
        pulled_arm, _ = oracolo.fit(self.budget, 100, 2)
        return pulled_arm

    def update_observations(self, reward, activated_edges, seen_edges):
        activated_edges = activated_edges
        self.collected_rewards.append(reward)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                arm = np.atleast_2d(self.feature_matrix_edges[i, j].T)
                if seen_edges[i,j] != 0:
                    self.M = self.M + self.sigma ** (-2) * np.dot(arm.T, arm)
                    if activated_edges[i, j] > 0:
                        self.B = self.B + arm.T


