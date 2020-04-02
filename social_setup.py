import numpy as np

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
