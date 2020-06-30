import numpy as np
from tqdm import trange

from social_influence import helper
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from social_influence.influence_maximisation import GreedyLearner
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.social_setup import SocialNetwork
from social_influence.const import FEATURE_MAX, FEATURE_PARAM
from social_influence.LinUCB.LinUCBLearner import *
from social_influence.LinUCB.LinUCBEnviroment import *

n_nodes = 100
c = 2
n_features = 5
T = 200
n_experiment = 10
budget = 2
n_steps = 2
mc_simulations = 5
social_name = "email_fixed"

helper = helper.Helper()
facebook = helper.read_dataset(social_name)
social_network = SocialNetwork(facebook, FEATURE_PARAM[0], FEATURE_MAX, max_nodes=n_nodes)

prob_matrix = social_network.get_matrix()
features_edge_matrix = social_network.get_edge_features_matrix()
n_edges = features_edge_matrix.shape[0]

reward_per_experiment = []
regret_per_experiment = []
tetha = []
env = LinUCBEnviroment(prob_matrix)
sampler = MonteCarloSampling(social_network.get_matrix())

opt, opt_seeds = env.opt(budget, mc_simulations, n_steps, parallel=True)

for e in trange(n_experiment):
    learner = LinUCBLearner(features_edge_matrix, mc_simulations, n_steps, budget, 2)
    regret_per_timestep = []
    cumulative_regret = 0

    for t in range(T):
        pulled_arm = learner.pull_arm()
        # reward = env.round(pulled_arm)
        learner_seeds, _ = learner.find_best_seeds(parallel=True)

        history_vector = sampler.simulate_episode(opt_seeds, n_steps)
        opt_reward = np.sum(history_vector) - budget

        history_vector, target_activated = sampler.simulate_episode(learner_seeds, n_steps, target_edge=pulled_arm)
        learner_reward = np.sum(history_vector) - budget
        learner.update_values(pulled_arm, int(target_activated))

        inst_regret = opt_reward-learner_reward

        cumulative_regret += inst_regret
        regret_per_timestep.append(cumulative_regret)

    regret_per_experiment.append(regret_per_timestep)

timesteps = []
results = []
indexes = []

for experiment, index in zip(regret_per_experiment, range(len(regret_per_experiment))):
    timesteps.extend(np.arange(len(experiment)))
    results.extend(experiment)
    indexes.extend([index] * len(experiment))

plt.figure()
df = pd.DataFrame({"regret": results, "timestep": timesteps, "experiment_id": indexes})
sns.lineplot(x="timestep", y="regret", data=df)
plt.title(social_name + ": mean regret over time")
plt.savefig("testLinUCB_" + social_name + ".png")
plt.show()