import numpy as np

class LinearMabEnvironment():
    def __init__(self, n_arms, dim): #dim is the dimension of the feature vector
        super().__init__()

        #theta are the parameters used to compute the probabilities of drawing the rewards
        self.theta = np.random.dirichlet(np.ones(dim), size=1)
        self.arms_features = np.random.binomial(1, 0.5, size=(n_arms,dim)) #rows are arms, columns are feature
        self.p = np.zeros(n_arms)

        #compute the probabilities for each arm
        for i in range(0,n_arms):
            self.p[i] = np.dot(self.theta, self.arms_features[i])
    
    #return the value returned by a bernoulli distribution
    def round(self, pulled_arm):
        return 1 if np.random.random() < self.p[pulled_arm] else 0

    #this function returns the maximum value that we will use to compute the regret
    def opt(self):
        return np.max(self.p)

