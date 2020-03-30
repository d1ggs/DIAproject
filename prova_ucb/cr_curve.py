import numpy as np
import matplotlib.pyplot as plt


class Product1_Season1():
    def __init__(self):
        self.points = np.array(
            [[0, 0.9], [0.5, 0.85], [1, .75], [1.5, .7], [2, .55], [2.5, .3], [3, .1], [3.5, .05], [4, .01], [4.5, 0],
             [5, 0], [5.5, 0], [6, 0], [7, 0]])
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = np.polyfit(self.x, self.y, 6)
        self.p = np.poly1d(self.z)

    def getProbability(self, price: int):
        if (price < 0 or price > 4):
            return 0
        elif self.p(price) > 1:
            return 1
        else:
            return self.p(price)

    def plot(self):
        new_x = np.linspace(0, 6, 100)
        plt.figure("Product1 season1")
        plt.plot(new_x, self.p(new_x), 'r')
        plt.show()