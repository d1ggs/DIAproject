import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
from social_influence.LinUCB.LinUCBEnvironment import LinUCBEnvironment
from social_influence.const import FEATURE_MAX, FEATURE_PARAM, SOCIAL_NAMES
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.budget_allocation import CumulativeBudgetAllocation, StatelessBudgetAllocation

from social_influence.LinUCB.LinUCBLearner import LinUCBLearner

from utils import seeds_to_binary

MAX_NODES = 300
TOTAL_BUDGET = 5
MAX_PROPAGATION_STEPS = 2
N_EXPERIMENTS = 10
T = 50
N_SOCIAL_NETWORKS = 3
MC_SIMULATIONS = 10
EXPLORATION_C_UCB = [2, 1, 5]
SAVEDIR = "./plots/ex_4/"

# Support variables
social_networks = []
enviroments = []
original_learners = []
samplers = []
products = []
regret_per_experiment = []

# Instantiate social networks and Monte-Carlo samplers
for i in range(N_SOCIAL_NETWORKS):
    social_network = SOCIAL_NAMES[i]
    helper = Helper()
    dataset = helper.read_dataset(social_network + "_fixed")

    social = SocialNetwork(dataset, FEATURE_PARAM[i], FEATURE_MAX, max_nodes=MAX_NODES)
    social_networks.append(social)

    sampler = MonteCarloSampling(social.get_matrix())
    samplers.append(sampler)

# Instantiate budget allocator and precompute optimal social influence
budget_allocator = CumulativeBudgetAllocation(social_networks[0].get_matrix(), social_networks[1].get_matrix(),
                                              social_networks[2].get_matrix(), TOTAL_BUDGET, MC_SIMULATIONS,
                                              MAX_PROPAGATION_STEPS)
budgets, opt, optimal_seeds = budget_allocator.joint_influence_maximization()

# Convert seeds from indices to binary vector
optimal_seeds = seeds_to_binary(optimal_seeds, MAX_NODES)

# Instantiate learners and their respective environments
for i in range(N_SOCIAL_NETWORKS):
    enviroments.append(LinUCBEnvironment(social_networks[i].get_matrix()))
    original_learners.append(LinUCBLearner(social_networks[i].get_edge_features_matrix(),
                                           MC_SIMULATIONS, MAX_PROPAGATION_STEPS, budgets[i], EXPLORATION_C_UCB[i]))

# Instantiate the budget allocator
comb_budget_allocator = StatelessBudgetAllocation(TOTAL_BUDGET, MC_SIMULATIONS, MAX_PROPAGATION_STEPS)

# Experiments loop
for e in range(N_EXPERIMENTS):
    print("\nExperiment {} of {}".format(e + 1, N_EXPERIMENTS))

    # Reset the learners
    learners = copy.deepcopy(original_learners)

    cumulative_regret = [0, 0, 0]
    regret_per_timestep = [np.asarray([0, 0, 0])]

    # Passing time loop
    for t in trange(T):

        for j in range(N_SOCIAL_NETWORKS):
            # Select candidate edge
            pulled_arm = learners[j].pull_arm()

            # Check if the edge would activate
            reward = enviroments[j].round(pulled_arm)

            # Update the edge
            learners[j].update_values(pulled_arm, reward)

        # Compute best seeds over the estimated matrices
        _, _, seeds = comb_budget_allocator.joint_influence_maximization(learners[0].get_prob_matrix(),
                                                                         learners[1].get_prob_matrix(),
                                                                         learners[2].get_prob_matrix())

        seeds = seeds_to_binary(seeds, MAX_NODES)

        comb_reward = 0
        clairvoyant_rew = 0

        random_seed = np.random.randint(0, 1000000)

        # Compute social influence regret for the social networks
        social_regrets = []
        for z in range(N_SOCIAL_NETWORKS):
            mc_sampler = samplers[z]

            # Compute activated nodes for clairvoyant algorithm
            clairvoyant_perf = np.sum(
                mc_sampler.simulate_episode(optimal_seeds[z], MAX_PROPAGATION_STEPS, random_seed=random_seed))

            # Compute acrivated nodes for estimated matrices
            active_nodes = mc_sampler.simulate_episode(seeds[z], MAX_PROPAGATION_STEPS, random_seed=random_seed)
            learner_perf = np.sum(active_nodes)

            # Cumulate the rewards for each learner
            comb_reward += learner_perf
            clairvoyant_rew += clairvoyant_perf
            regret = clairvoyant_perf - learner_perf
            cumulative_regret[z] += regret
            social_regrets.append(cumulative_regret[z])

        # Compute the regret
        # cumulative_regret += clairvoyant_rew - comb_reward
        regret_per_timestep.append(np.asarray(social_regrets))

    regret_per_experiment.append(np.asarray(regret_per_timestep))

# Plot results

timesteps = []
results = []
indexes = []

# Cumulative plot
print(regret_per_experiment)
regret_per_experiment = np.asarray(regret_per_experiment)
total_regret = np.sum(regret_per_experiment, axis=2)

for experiment, index in zip(total_regret, range(len(total_regret))):
    timesteps.extend(np.arange(len(experiment)))
    results.extend(experiment)
    indexes.extend([index] * len(experiment))

plt.figure()
df = pd.DataFrame({"regret": results, "timestep": timesteps, "experiment_id": indexes})
sns.lineplot(x="timestep", y="regret", data=df)
plt.title("mean regret over time")
plt.savefig(SAVEDIR + "ex_4_cumulative_{}n_{}bdg_{}prop_{}ex.png".format(MAX_NODES, TOTAL_BUDGET,
                                                                         MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
plt.show()

# Single social plot
for social_network, index in zip(SOCIAL_NAMES, range(len(SOCIAL_NAMES))):
    timesteps = []
    results = []
    indexes = []

    for experiment, exp_index in zip(regret_per_experiment, range(len(regret_per_experiment))):
        data = experiment[:, index]
        # Prepare the data structures for the dataframe
        timesteps.extend(np.arange(len(data)))
        results.extend(data)
        indexes.extend([exp_index] * len(data))

    plt.figure()
    df = pd.DataFrame({"regret": results, "timestep": timesteps, "experiment_id": indexes})
    sns.lineplot(x="timestep", y="regret", data=df)
    plt.title(social_network + " : mean regret over time")
    plt.savefig(SAVEDIR + "ex_4_{}_{}n_{}bdg_{}prop_{}ex.png".format(social_network, MAX_NODES, TOTAL_BUDGET,
                                                                     MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
    plt.show()

