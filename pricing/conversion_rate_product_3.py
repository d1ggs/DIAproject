import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
import math
from pricing.conversion_rate import ConversionRateCurve


class Product2Season1(ConversionRateCurve):
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
        plt.figure("Product2 season1")
        plt.title("Product2 season1")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


class Product2Season2(ConversionRateCurve):
    def __init__(self, n_prices):
        self.x = np.linspace(0, n_prices, 14)
        self.y = np.array([0.95, 0.60, 0.40, .37, .34, .3, .3, .27, .25, .20, .22, .15, .1, .1])
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
        plt.figure("Product2 season2")
        plt.title("Product2 season2")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


class Product2Season3(ConversionRateCurve):
    def __init__(self, n_prices):
        self.x = np.linspace(0, n_prices, 14)
        self.y = [1., 0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.3, 0.1, 0., 0., 0.]

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
        plt.figure("Product2 season3")
        plt.title("Product2 season3")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


if __name__ == '__main__':
    a = Product2Season1(2)
    b = Product2Season2(4)
    c = Product2Season3(7)

    a.plot()
    b.plot()
    c.plot()
