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

    def __init__(self, n_prices):
        self.x = np.linspace(0, n_prices, 14)
        self.y = [0.9, 0.85, 0.75, 0.7, 0.55, 0.3, 0.1, 0.05, 0.01, 0., 0., 0., 0., 0.]
        self.z = np.polyfit(self.x, self.y, 6)
        self.p = np.poly1d(self.z)
        self.n_prices = n_prices

    def get_probability(self, arm: int):
        prob = self.p(arm)
        if arm < 0 or arm > self.n_prices:
            return 0
        elif prob > 1:
            return 1
        elif prob < 0:
            return 0
        else:
            return prob

    def plot(self):
        new_x = np.linspace(0, self.n_prices, 100)
        plt.figure("Product1 season1")
        plt.title("Product1 season1")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


class Product1Season2(ConversionRateCurve):
    def __init__(self, n_prices):
        self.x = np.linspace(0, n_prices, 14)

        self.y = [1., 0.99, 0.95, 0.9, 0.85, 0.8, 0.77, 0.65, 0.45, 0.3, 0.1, 0., 0., 0.]

        self.z = np.polyfit(self.x, self.y, 6)
        self.p = np.poly1d(self.z)
        self.n_prices = n_prices

    def get_probability(self, arm: int):
        prob = self.p(arm)
        if arm < 0 or arm > self.n_prices:
            return 0
        elif prob > 1:
            return 1
        elif prob < 0:
            return 0
        else:
            return prob

    def plot(self):
        new_x = np.linspace(0, self.n_prices, 100)
        plt.figure("Product1 season2")
        plt.title("Product1 season2")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


class Product1Season3(ConversionRateCurve):
    def __init__(self, n_prices):
        self.x = np.linspace(0, n_prices, 14)
        self.y = [1., 0.9, 0.85, 0.75, 0.6, 0.4, 0.2, 0.1, 0.05, 0., 0., 0., 0., 0.]

        self.z = np.polyfit(self.x, self.y, 6)
        self.p = np.poly1d(self.z)
        self.n_prices = n_prices


    def get_probability(self, arm: int):
        prob = self.p(arm)
        if arm < 0 or arm > self.n_prices:
            return 0
        elif prob > 1:
            return 1
        elif prob < 0:
            return 0
        else:
            return prob


    def plot(self):
        new_x = np.linspace(0, self.n_prices, 100)
        plt.figure("Product1 season3")
        plt.title("Product1 season3")
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
    a = Product1Season1(6)
    b = Product1Season2(6)
    c = Product1Season3(6)
    a.plot()
    b.plot()
    c.plot()
