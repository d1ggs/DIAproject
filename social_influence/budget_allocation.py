import numpy as np
from pulp import *
from social_influence.influence_maximisation import GreedyLearner

class GreedyBudgetAllocation:

    def __init__(self, social1, social2, social3, budget_total,  mc_simulations, n_steps_montecarlo):

        """
        In the main we need to pass 3 social media objects through the main
        Parameters
        ----------
        social1
        social2
        social3
        """
        self.social1_learner = GreedyLearner(social1.get_matrix(), social1.get_n_nodes())
        self.social2_learner = GreedyLearner(social2.get_matrix(), social2.get_n_nodes())
        self.social3_learner = GreedyLearner(social3.get_matrix(), social3.get_n_nodes())
        assert(budget_total >= 3)
        self.budget_total = budget_total
        self.max_budget = np.sum(self.budget)
        self.mc_simulations = mc_simulations
        self.n_steps_montecarlo = n_steps_montecarlo

    def joint_influence_calculation(self, budget: list):
        seeds1, influence1 = self.social1_learner.fit(self.mc_simulations, self.n_steps_montecarlo, budget[0])
        seeds2, influence2 = self.social1_learner.fit(self.mc_simulations, self.n_steps_montecarlo, budget[1])
        seeds3, influence3 = self.social1_learner.fit(self.mc_simulations, self.n_steps_montecarlo, budget[2])
        return influence1 + influence2 + influence3


    def joint_influence_maximization(self):

        prob = LpProblem("Budget allocation through social media", LpMaximize)
        # Initial budget distribution, given that budget is greater than 2 we could have this distribution (we could randomize it)
        budget = list([self.budget_total-2, 1, 1])
        budget_0 = LpVariable("Budget_0", lowBound=1, upBound=self.budget_total-2, cat="Integer")
        budget_1 = LpVariable("Budget_1", lowBound=1, upBound=self.budget_total-2, cat="Integer")
        budget_2 = LpVariable("Budget_2", lowBound=1, upBound=self.budget_total-2, cat="Integer")

        prob += self.joint_influence_calculation([value(budget_0), value(budget_1), value(budget_2)])



        # Define constraint
        prob += budget_0 + budget_1 + budget_2 == self.budget_total
        # We utilize cvxpy to do integer programming, with

        status = prob.solve()
        print("Status: ", LpStatus(status))

        for v in prob.variables():
            if v.varValue > 0:
                print(v.name, "=", v.varValue)
        # TODO dio se non me la mandi buona ritorno credente e sono cazzi tuoi poi perche' mi becchi in paradiso






        




