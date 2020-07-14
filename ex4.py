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
T = 20



# Simulate Social Network
n_social_networks = 3
social_networks = []
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

budget_allocator = CumulativeBudgetAllocation(social_networks[0].get_matrix(), social_networks[1].get_matrix(),
                                              social_networks[2].get_matrix(), TOTAL_BUDGET, monte_carlo_simulations,
                                              n_steps_max)
budgets, _, _ = budget_allocator.joint_influence_maximization()

envs = []
opts = []
opt_seeds = []
learners = []
regret_per_experiment = []

for i in range(n_social_networks):
    enviroment = LinUCBEnviroment(social_networks[i].get_matrix())
    opt, opt_seed = enviroment.opt(budgets[i], monte_carlo_simulations, n_steps_max, parallel=True)
    sampler = MonteCarloSampling(social_networks[i].get_matrix())
    envs.append(enviroment)
    opts.append(opt)
    opt_seeds.append(opt_seed)
    samplers.append(sampler)

regret_per_timestep = [[], [], []]
regret_per_experiment = [[], [], []]

for i in trange(n_social_networks):
    for e in range(N_EXPERIMENTS):
        learner = LinUCBLearner(social_networks[i].get_edge_features_matrix(), monte_carlo_simulations, n_steps_max,
                                budgets[i], c[i])
        cumulative_regret = 0

        for t in range(T):
            pulled_arm = learner.pull_arm()
            # reward = env.round(pulled_arm)
            learner_seeds, _ = learner.find_best_seeds(parallel=True)

            history_vector = sampler.simulate_episode(opt_seeds[i], n_steps_max)
            opt_reward = np.sum(history_vector) - budgets[i]

            history_vector, target_activated = sampler.simulate_episode(learner_seeds, n_steps_max,
                                                                        target_edge=pulled_arm)
            learner_reward = np.sum(history_vector) - budgets[i]
            learner.update_values(pulled_arm, int(target_activated))

            inst_regret = opt_reward - learner_reward

            cumulative_regret += inst_regret
            regret_per_timestep[i].append(cumulative_regret)

        regret_per_experiment[i].append(regret_per_timestep)

total_regret_per_timestep = regret_per_timestep[0] + regret_per_timestep[1] + regret_per_timestep[2]
total_regret_per_experiment = regret_per_experiment[0] + regret_per_experiment[1] + regret_per_experiment[2]

timesteps = []
results = []
indexes = []

for experiment, index in zip(total_regret_per_experiment, range(len(total_regret_per_experiment))):
    timesteps.extend(np.arange(len(experiment)))
    results.extend(experiment)
    indexes.extend([index] * len(experiment))

plt.figure()
df = pd.DataFrame({"regret": results, "timestep": timesteps, "experiment_id": indexes})
sns.lineplot(x="timestep", y="regret", data=df)
plt.title("mean regret over time")
plt.savefig("LinUCB" ".png")
plt.show()
