import numpy as np
from tqdm import trange

from social_influence import helper
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from social_influence.influence_maximisation import GreedyLearner
from social_influence.social_setup import SocialNetwork
from social_influence.const import FEATURE_MAX, FEATURE_PARAM
from social_influence.LinUCB.InfluenceMaximizator import *
from social_influence.LinUCB.LinUCBLearner import *
from social_influence.LinUCB.LinUCBEnviroment import *

n_nodes = 200
c = 2
n_features = 5
T = 1000
n_experiment = 10
budget = 5
n_steps = 3
mc_simulations = 100
helper = helper.Helper()
facebook = helper.read_dataset("gplus_fixed")
social_network = SocialNetwork(facebook, FEATURE_PARAM[0], FEATURE_MAX, max_nodes=n_nodes)

prob_matrix = social_network.get_matrix()
features_edge_matrix = social_network.get_edge_features_matrix()
n_edges = features_edge_matrix.shape[0]

reward_per_experiment = []
regret_per_experiment = []
tetha = []
env = LinUCBEnviroment(prob_matrix)
maximizator = InfluenceMaximizator(features_edge_matrix, n_experiment, budget, mc_simulations, n_steps)

for e in trange(n_experiment):
    learner = LinUCBLearner(features_edge_matrix,c)
    for t in range(T):
        pulled_arm = learner.pull_arm()
        reward = env.round(pulled_arm)
        learner.update_values(pulled_arm, reward)
    maximizator.update_tetha(learner.get_theta())
    reward_per_experiment.append(learner.collected_rewards)


opt = env.opt()


plt.figure(0)
plt.title("LinearUCB\n")
plt.ylabel("Regret")
plt.xlabel("T")
plt.plot(np.cumsum(np.mean(opt - reward_per_experiment, axis=0)), 'r')
plt.savefig("Pippo.png")
plt.plot(opt, 'g')
plt.show()

best_seed_approx,approx_reward = maximizator.find_best_seeds()
print("Best seed with approximate matrix is: {}. Reward: {}\n".format(best_seed_approx,approx_reward))

greedy_learner = GreedyLearner(prob_matrix, n_nodes)
best_seed, reward = greedy_learner.parallel_fit(budget, mc_simulations, n_steps)
print("Best seed with true matrix is: {}. Reward: {}\n".format(best_seed,reward))