import numpy as np
import pandas as pd
import os

from const import ROOT_PROJECT_PATH, MATRIX_PATH

class SocialNetwork:
    def __init__(self,dataset,parameters, feature_max):
        '''
        Features in a Social Network are: Tag, Share, Like, Message, Comment
        They are saved in self.features as a numpy array, ordered as written above (self.features[0] -> Tag...)
        '''
        
        self.social_edges, self.features = dataset
        self.parameters = parameters
        self.feature_max = feature_max

        assert(self.parameters.shape == self.features[0].shape)
        assert(np.sum(self.parameters)==1)
        self.matrix = self.probability_matrix()

    def compute_activation_prob(self,features):
        out = np.dot(self.parameters,features) #dot product
        prob = out/self.feature_max  #divide by the maximum value of a feature
        return prob

    def probability_matrix(self):
        max_node = self.social_edges.max()
        matrix = np.zeros((max_node+1,max_node+1))
        
        for i in range(self.social_edges.shape[0]):
            node_a = self.social_edges[i][0]
            node_b = self.social_edges[i][1]
            features = self.features[i]
            matrix[node_a,node_b] = self.compute_activation_prob(features)

        
        ##np.save(os.path.join(ROOT_PROJECT_PATH, MATRIX_PATH),matrix)
        return matrix
    
    def get_matrix(self):
        return self.matrix



