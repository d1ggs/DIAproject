import numpy as np
import os

from social_influence.const import ROOT_PROJECT_PATH, MATRIX_PATH, FEATURE_MAX
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.influence_maximisation import InfluenceLearner


if __name__ == "__main__": 
    
    #Simulate Social Network
    parameters = np.array([[0.1, 0.3, 0.2, 0.2,0.2],[0.4, 0.1, 0.2, 0.2,0.1],[0.5, 0.1, 0.1, 0.1,0.2]]) #parameters for each social

    # matrix_path = os.path.join(ROOT_PROJECT_PATH,MATRIX_PATH)
    # if os.path.isfile(matrix_path):
    #     prob_matrix = np.load(matrix_path)
    # else:
    
    helper = Helper()
    dataset = helper.read_dataset()
    social = SocialNetwork(dataset,parameters[0], FEATURE_MAX)
    prob_matrix = social.get_matrix()
    n_nodes = prob_matrix.shape[0]
    
    #n_nodes = 50
    #prob_matrix = np.random.uniform(0.0,0.01,(n_nodes,n_nodes))

    influence_learner = InfluenceLearner(prob_matrix)
    
    n_episodes = 10
    seeds = np.random.binomial(1, 0.1, size=(n_nodes))
    
    nodes_probabilities = influence_learner.mc_sampling(seeds,n_episodes)
    print(', '.join([str(d) for d in nodes_probabilities]))