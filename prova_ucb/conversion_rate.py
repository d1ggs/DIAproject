from abc import ABC
import math
import numpy as np
import matplotlib.pyplot as plt


class ConversionRateCurve(ABC):
    def get_probability(self, x):
        pass


class Logistic(ConversionRateCurve):
    def __init__(self, mid: float, growth=1):
        """:param mid: the point where the curve outputs 0.5"""
        self.mid = mid
        self.growth = growth

    def get_probability(self, x):
        res = 1 - 1 / (self.growth * (1 + math.e ** (-x + self.mid)))
        return res


class Linear(ConversionRateCurve):
    def __init__(self, y0, m):
        self.y0 = y0
        self.m = m

    def get_probability(self, x):
        return self.m * x + self.y0


class Product1Season1(ConversionRateCurve):
    def __init__(self):
        self.points = np.array(
            [[0, 0.9], [0.5, 0.85], [1, .75], [1.5, .7], [2, .55], [2.5, .3], [3, .1], [3.5, .05], [4, .01], [4.5, 0],
             [5, 0], [5.5, 0], [6, 0], [7, 0]])
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = np.polyfit(self.x, self.y, 6)
        self.p = np.poly1d(self.z)

    def get_probability(self, price: int):
        prob = self.p(price)
        if price < 0 or price > 4:
            return 0
        elif prob > 1:
            return 1
        elif prob < 0:
            return 0
        else:
            return prob

    def plot(self):
        new_x = np.linspace(0, 6, 100)
        plt.figure("Product1 season1")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


class Product1Season2(ConversionRateCurve):
    def __init__(self):
        self.points = np.array(
            [[0, 1], [0.5, 0.99], [1, .95], [1.5, .9], [2, .85], [2.5, .8], [3, .77], [3.5, .65], [4, .45], [4.5, .30],
             [5, .1], [5.5, 0], [6, 0], [6.5, 0], [7, 0]])
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = np.polyfit(self.x, self.y, 6)
        self.p = np.poly1d(self.z)

    def get_probability(self, price: int):
        prob = self.p(price)
        if price < 0 or price > 4:
            return 0
        elif prob > 1:
            return 1
        elif prob < 0:
            return 0
        else:
            return prob

    def plot(self):
        new_x = np.linspace(0, 6, 100)
        plt.figure("Product1 season2")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


class Product1Season3(ConversionRateCurve):
    def __init__(self):
        self.points = np.array(
            [[0, 1], [0.5, 0.9], [1, .85], [1.5, .75], [2, .60], [2.5, .40], [3, .20], [3.5, .10], [4, .05], [4.5, .0],
             [5, 0], [5.5, 0], [6, 0], [6.5, 0], [7, 0]])
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = np.polyfit(self.x, self.y, 6)
        self.p = np.poly1d(self.z)

    def get_probability(self, price: int):
        prob = self.p(price)
        if price < 0 or price > 6:
            return 0
        elif prob > 1:
            return 1
        elif prob < 0:
            return 0
        else:
            return prob

    def plot(self):
        new_x = np.linspace(0, 7, 100)
        plt.figure("Product1 season2")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()



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
