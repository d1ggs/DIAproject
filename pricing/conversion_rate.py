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


class ProductConversionRate(ConversionRateCurve):
    def __init__(self, product_id: int, season: int, n_prices: int, y_values: list):
        self.id = product_id
        self.season = season
        self.x = np.linspace(0, n_prices, 14)
        self.y = y_values
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
        plt.figure()
        plt.title("Product {} season {}".format(self.id, self.season))
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()