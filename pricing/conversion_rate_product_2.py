import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
import math
from pricing.conversion_rate import ConversionRateCurve


class Product2Season1(ConversionRateCurve):
    def __init__(self):
        self.points = np.array(
            [[0, 0.95], [0.5, 0.60], [1, .40], [1.5, .37], [2, .34], [2.5, .3], [3, .3], [3.5, .27], [4, .25], [4.5, .20],
             [5, .22], [5.5,.15], [6, .1], [7, .1]])
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
        new_x = np.linspace(0, 6, 100)
        plt.figure("Product1 season1")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()


class Product2Season2(ConversionRateCurve):
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
        if price < 0 or price > 6:
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


class Product2Season3(ConversionRateCurve):
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
        new_x = np.linspace(0, 6, 100)
        plt.figure("Product1 season2")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()

if __name__ == '__main__':
    a = Product2Season1()
    b = Product2Season2()
    c = Product2Season3()

    a.plot()
    b.plot()
    c.plot()

