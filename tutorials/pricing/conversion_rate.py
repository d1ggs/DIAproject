from abc import ABC
import math


class ConversionRateCurve(ABC):
    def compute(self, x):
        pass


class Logistic(ConversionRateCurve):
    def __init__(self, mid: float, growth=1):
        self.mid = mid
        self.growth = growth

    def compute(self, x):
        res = 1 - 1 / (self.growth * (1 + math.e ** (-x + self.mid)))
        return res


class DemandModel(object):
    def __init__(self, cr_curve):
        self.cr_curve = cr_curve

    def compute_buyers(self, n_users, price):
        return round(n_users * self.cr_curve.compute(price))


if __name__ == '__main__':
    results = []
    for i in range(1, 6):
        fun = Logistic(50, i)
        env = DemandModel(fun)
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
