from tqdm import trange

from social_influence import helper
from social_influence.IMLinUCB.IMLinUCBEnviroment import IMLinUCBEnviroment
from social_influence.IMLinUCB.IMLinUCBLearner import IMLinUCBLearner
from social_influence.IMLinUCB.create_dataset import create_dataset2
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from social_influence.social_setup import SocialNetwork
from social_influence.const import FEATURE_MAX, FEATURE_PARAM

import numpy as np

budget = 2
mc_simulations_opt = 5
mc_simulations = 5
n_steps = 2
n_nodes = 30
n_features = 5
T = 400
n_experiment = 1
# parameters = np.asarray(
#         ((0.1, 0.3, 0.2, 0.2, 0.2), (0.3, 0.1, 0.2, 0.2, 0.2), (0.5, 0.1, 0.1, 0.1, 0.2)))  # parameters for each social

helper = helper.Helper()
facebook = helper.read_dataset("gplus_fixed")
social_network = SocialNetwork(facebook, FEATURE_PARAM[0], FEATURE_MAX, max_nodes=n_nodes)

# prob_matrix, features_edge_matrix, n_edges = create_dataset2(n_nodes, n_features, parameters)
prob_matrix = social_network.get_matrix()
features_edge_matrix = social_network.get_edge_features_matrix()
n_edges = features_edge_matrix.shape[0]

reward_per_experiment = []
regret_per_experiment = []

env = IMLinUCBEnviroment(prob_matrix, budget, mc_simulations, n_steps)
optimal_reward, best_seed = env.opt(parallel=True)
print("opt", optimal_reward)

for exp in range(n_experiment):
    #  env = IMLinUCBEnviroment(prob_matrix, budget, n_steps)
    learner = IMLinUCBLearner(n_features, features_edge_matrix, budget, n_nodes, mc_simulations, n_steps, T)
    # optimal_reward = env.opt()
    cumulative_regret = 0
    regret_per_timestep = []
    count = 0
    for t in trange(T):
        pulled_arm = learner.pull_arm()
        # print(pulled_arm)
        # print(best_seed)
        # print("--------------------")
        reward, edge_activation_matrix, seen_edges = env.round(pulled_arm)
        # comparison = pulled_arm == best_seed
        # equal_arrays = comparison.all()
        # print(equal_arrays)
        # if equal_arrays:
        #     count += 1
        #     print("percentage:", count / (t+1.0))
        inst_regret = optimal_reward - reward
        # assert not (inst_regret < 0), "Negative regret"
        cumulative_regret += inst_regret
        learner.update_observations(reward, edge_activation_matrix, seen_edges)

        regret_per_timestep.append(cumulative_regret)

    reward_per_experiment.append(learner.collected_rewards)
    regret_per_experiment.append(regret_per_timestep)
    # plt.figure()
    # plt.plot(regret_per_timestep, 'g')
    # plt.show()

# sistemo sintassi
# another_env = IMLinUCBEnviroment(prob_matrix, budget, n_steps)
# optimal_reward = another_env.opt()

# plt.figure()
# plt.xlabel("Time")
# plt.ylabel("Regret")
# plt.plot(np.mean(regret_per_experiment, axis=0), 'g')
# plt.savefig("plot.png")

experiments = range(n_experiment)
indexes = []
regrets = []
timesteps = []

for r, e in zip(regret_per_experiment, experiments):
    indexes.extend([e] * len(r))
    regrets.extend(r)
    timesteps.extend(list(range(len(r))))

data = {"exp_id": indexes,
        "cumulative_regret": regrets,
        "timestep": timesteps}

df = pd.DataFrame(data)
# print(df)

plt.figure()
sns.lineplot(x="timestep", y="cumulative_regret", data=df)
plt.show()
plt.savefig("plot_seaborn.png")
