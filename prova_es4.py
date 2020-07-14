import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
import copy
from social_influence.LinUCB.LinUCBEnviroment import LinUCBEnviroment
from pricing.conversion_rate import ProductConversionRate
from pricing.environments import StationaryEnvironment, NonStationaryEnvironment
from pricing.learners.ts_learner import TSLearner
from pricing.const import TIME_HORIZON, N_EXPERIMENTS, PRICES, N_ARMS
from social_influence.const import FEATURE_MAX, FEATURE_PARAM, SOCIAL_NAMES
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.budget_allocation import CumulativeBudgetAllocation

from social_influence.LinUCB.LinUCBLearner import LinUCBLearner
MAX_NODES = 50
TOTAL_BUDGET = 7
MAX_PROPAGATION_STEPS = 2
N_EXPERIMENTS = 5
T = 10


n_social_networks = 3
social_networks = []
enviroments = []
learners = []
samplers = []
products = []
monte_carlo_simulations = 10
n_steps_max = 2
c = [2, 1, 5]

for i in range(n_social_networks):
    social_network = SOCIAL_NAMES[i]
    helper = Helper()
    dataset = helper.read_dataset(social_network + "_fixed")

    social = SocialNetwork(dataset, FEATURE_PARAM[i], FEATURE_MAX, max_nodes=MAX_NODES)
    social_networks.append(social)
    sampler = MonteCarloSampling(social.get_matrix())
    samplers.append(sampler)

budget_allocator = CumulativeBudgetAllocation(social_networks[0].get_matrix(), social_networks[1].get_matrix(),
                                              social_networks[2].get_matrix(), TOTAL_BUDGET, monte_carlo_simulations,
                                              n_steps_max)
budgets, opt, _ = budget_allocator.joint_influence_maximization()

for i in range(n_social_networks):
    enviroments.append(LinUCBEnviroment(social_networks[i].get_matrix()))
    learners.append(LinUCBLearner(social_networks[i].get_edge_features_matrix(), monte_carlo_simulations,n_steps_max, budgets[i] , c[i]))


regret_per_experiment = [[], [], []]
reward_per_timestep = []


for e in range(N_EXPERIMENTS):
    for i in range(n_social_networks):
        learners.append(
            LinUCBLearner(social_networks[i].get_edge_features_matrix(), monte_carlo_simulations, n_steps_max,
                          budgets[i], c[i]))
    cumulative_regret = 0
    regret_per_timestep = []
    for t in range(T):
        for j in range(n_social_networks):
            pulled_arm = learners[i].pull_arm()
            reward = enviroments[i].round(pulled_arm)
            learners[i].update_values(pulled_arm,reward)
        comb_budget_allocator=CumulativeBudgetAllocation(learners[0].get_prob_matrix(),learners[1].get_prob_matrix(),learners[2].get_prob_matrix(),TOTAL_BUDGET,monte_carlo_simulations,n_steps_max)
        _, _, seeds = comb_budget_allocator.joint_influence_maximization()
        comb_reward = 0
        for z in range(n_social_networks):
            mc_sampler = MonteCarloSampling(learners[i].get_prob_matrix())
            seed = np.zeros(MAX_NODES)
            seed[seeds[z]] = 1
            comb_reward += np.sum(mc_sampler.simulate_episode(seed, n_steps_max))
        regret_per_timestep.append(opt-comb_reward)
    regret_per_experiment[i].append(regret_per_timestep)


"""
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
plt.title("mean regret over time")
plt.savefig("LinUCB" ".png")
plt.show()
"""

plt.figure(0)
plt.plot(np.cumsum(np.mean(regret_per_experiment,axis = 0)),'r')
plt.show()