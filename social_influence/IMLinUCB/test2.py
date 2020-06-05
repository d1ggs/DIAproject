from social_influence.IMLinUCB.IMLinUCBEnviroment import *
from social_influence.IMLinUCB.IMLinUCBLearner import *
from social_influence.IMLinUCB.create_dataset import *
import matplotlib.pyplot as plt

budget = 2
n_steps = 2
n_nodes = 20
n_features = 5
T = 30
n_experiment = 30
parameters = np.array([0.7, 0.01, 0.3, 0.1, 0.2])
prob_matrix, features_edge_matrix, n_edges = create_dataset2(n_nodes, n_features, parameters)
reward_per_experiment = []
regret_per_experiment = []

env = IMLinUCBEnviroment(prob_matrix, budget, n_steps)
optimal_reward = env.opt()

for exp in range(n_experiment):
  #  env = IMLinUCBEnviroment(prob_matrix, budget, n_steps)
    learner = IMLinUCBLearner(n_features, features_edge_matrix, budget, n_steps)
   # optimal_reward = env.opt()
    cumulative_regret = 0
    regret_per_timestep = []
    for t in range(T):
        pulled_arm = learner.pull_arm()
        reward, edge_activation_matrix, seen_edges = env.round(pulled_arm)
        inst_regret = optimal_reward - reward
        cumulative_regret += inst_regret
        learner.update_observations(reward, edge_activation_matrix, seen_edges)

        regret_per_timestep.append(cumulative_regret)

    reward_per_experiment.append(learner.collected_rewards)
    regret_per_experiment.append(regret_per_timestep)
    plt.figure()
    plt.plot(regret_per_timestep, 'g')
    plt.show()

# sistemo sintassi
another_env = IMLinUCBEnviroment(prob_matrix, budget, n_steps)
#optimal_reward = another_env.opt()

plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.plot(np.mean(regret_per_experiment, axis=0), 'g')
plt.show()
