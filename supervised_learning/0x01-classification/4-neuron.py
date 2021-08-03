#!/usr/bin/env python3
"""Beginning of neuron class"""


import numpy as np


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
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Logistic Regression Cost Function"""
        div_cost = -1/A.shape[1]
        costs = (Y * np.log(A)) + ((1.0000001 - Y) * np.log(1.0000001 - A))
        return np.sum(costs) * div_cost

    def evaluate(self, X, Y):
        """Evaluates the predictions made and the cost"""
        predictions = self.forward_prop(X)
        cost = self.cost(Y, predictions)
        eval_bool = predictions >= 0.5
        evaluation = eval_bool.astype(int)
        return evaluation, cost
