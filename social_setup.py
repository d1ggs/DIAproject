import numpy as np
from helper import Helper


class SocialNetwork:
    def __init__(self):
        '''
        Features in a Social Network are: Tag, Share, Like, Message, Comment
        They are saved in self.features as a numpy array, ordered as written above (self.features[0] -> Tag...)
        '''
        self.helper = Helper()
        self.social_edge1, self.social_edge2, self.social_edge3, self.features = self.helper.read_dataset()


    #TODO calcolo probabilità e matrici di adiacenza con probabilità
    @staticmethod
    def compute_activation_prob(features, feature_max, social_id):
        parameters = np.array([[0.1, 0.3, 0.2, 0.2,0.2],[0.4, 0.1, 0.2, 0.2,0.1],[0.5, 0.1, 0.1, 0.1,0.2]]) #parameters for each social
        out = np.dot(parameters[social_id],features) #dot product
        prob = out/feature_max  #divide by the maximum value of a feature
        return prob


feature_max = 5
social_id = 0

features = np.random.randint(0,feature_max, 5) 
print(features)
prob = compute_activation_prob(features, feature_max-1, social_id )
print(prob)
