from abc import ABC
import math
import numpy as np


class ConversionRateCurve(ABC):
    def compute(self, x):
        pass


class Logistic(ConversionRateCurve):
    def __init__(self, mid: float, growth=1):
        """:param mid: the point where the curve outputs 0.5"""
        self.mid = mid
        self.growth = growth

    def compute(self, x):
        res = 1 - 1 / (self.growth * (1 + math.e ** (-x + self.mid)))
        return res


class DemandModel(object):
    def __init__(self, cr_curve, n_arms):
        self.cr_curve = cr_curve
        self.n_arms = n_arms

    def compute_buyers(self, n_users, price):
        """Compute how many users will buy, sampling from a Bernoulli distribution following the conversion rate"""
        buyers = 0
        conversion_rate = self.cr_curve.compute(price)

        # Sample actual buyers from a binomial distribution
        for _ in range(n_users):
            buyers += np.random.binomial(1, conversion_rate)
        return buyers

    def optimal_choice(self):
        """Compute best arm"""
        return max([self.cr_curve.compute(i) for i in range(1, self.n_arms + 1)])


if __name__ == '__main__':
    results = []
    for i in range(1, 6):
        fun = Logistic(50, i)
        env = DemandModel(fun, 100)
        partial = []
        for j in range(100):
            partial.append(env.compute_buyers(100, j))
        results.append(partial)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.xlabel("Price")
    plt.ylabel("Converted users")

    for partial in results:
        plt.plot(partial)

    plt.legend(['1', '2', '3', '4'])
    plt.show()
