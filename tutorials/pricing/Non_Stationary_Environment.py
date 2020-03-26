from Environment import *


class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probalities, horizon):
        super().__init__(n_arms, probalities)
        self.t = 0
        self.horizon = horizon

    def round(self, pulled_arm):
        n_phases = len(self.probabilities)
        phase_size = self.horizon/n_phases
        current_phase = int(self.t/phase_size)

        p = self.probabilities[current_phase][pulled_arm]
        self.t+=1
        return np.random.binomial(1,p)