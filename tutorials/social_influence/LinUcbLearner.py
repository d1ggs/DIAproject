import numpy as np


#UCBLearner learns parameters theta by multiplying matrix M and vector B
class LinUcbLearner():
    def __init__(self, arms_features):
        super().__init__()
        self.arms = arms_features
        self.dim = arms_features.shape[1]
        self.collected_rewards = []   #store the value at each round
        self.pulled_arms =[]
        self.c = 2.0  #exploration factor (it is possible to use also other values)
        self.M = np.identity(self.dim)
        self.b = np.atleast_2d(np.zeros(self.dim)).T
        self.theta = np.dot(np.linalg.inv(self.M),self.b)

    """
    Compute the Upper confidence bound to select which arm to pull
    """
    def compute_ucbs(self):
        self.theta = np.dot(np.linalg.inv(self.M), self.b)
        ucbs = []
        for arm in self.arms:
            arm = np.atleast_2d(arm).T
            ucb = np.dot(self.theta.T, arm) + self.c * np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
            ucbs.append(ucb[0][0])
        return ucbs

    def pull_arm(self):
        ucbs = self.compute_ucbs()
        return np.argmax(ucbs) #index of the maximum value

    """
    Update the values of matrix M and vector b after getting a rewards
    """
    def update_estimation(self, arm_idx, reward):
        arm = np.atleast_2d(self.arms[arm_idx]).T
        self.M += np.dot(arm, arm.T)
        self.b += reward * arm

    #arm_idx is the arm just pulled
    def update(self, arm_idx, reward):
        self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx,reward)

