#!/usr/bin/env python3
"""Beginning of neuron class"""


import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """Neuron Class for ML"""

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for weigths vector"""
        return self.__W

    @property
    def b(self):
        """getter for biases"""
        return self.__b

    @property
    def A(self):
        """getter for activation value A"""
        return self.__A

    def forward_prop(self, X):
        """Logistic Regression Foward Propagation"""
        self.z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-self.z))
        return self.__A

    def cost(self, Y, A):
        """Logistic Regression Cost Function"""
        mth = -1/A.shape[1]
        costs = (Y * np.log(A)) + ((1.0000001 - Y) * np.log(1.0000001 - A))
        return np.sum(costs) * mth

    def evaluate(self, X, Y):
        """Evaluates the predictions made and the cost"""
        predictions = self.forward_prop(X)
        cost = self.cost(Y, predictions)
        eval_bool = predictions >= 0.5
        evaluation = eval_bool.astype(int)
        return evaluation, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient Descent Method"""
        mth = 1/A.shape[1]
        dz = A - Y
        dw = mth * np.matmul(X, dz.T)
        db = mth * np.sum(dz)
        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains model based on forward prop and back prop"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            costs, x_points = [], []
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        for x in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if (x == 0 or x % step == 0) and (verbose is True or graph is True):
                cost = self.cost(Y, self.A)
                costs.append(cost), x_points.append(x)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(x, cost))
        if verbose is True or graph is True:
            cost = self.cost(Y, self.A)
            costs.append(cost), x_points.append(iterations)
            if verbose is True:
                print("Cost after {} iterations: {}".format(iterations, cost))
        if graph is True:
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.suptitle("Training Cost")
            plt.plot(x_points, costs, "b")
            plt.show()
        return self.evaluate(X, Y)
