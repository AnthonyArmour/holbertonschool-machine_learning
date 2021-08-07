#!/usr/bin/env python3
"""Neural Network Module"""


import numpy as np


class NeuralNetwork():
    """Neural Network Class"""

    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros([nodes, 1])
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for weigths_1 vector"""
        return self.__W1

    @property
    def b1(self):
        """getter for biases_1 vector"""
        return self.__b1

    @property
    def A1(self):
        """getter for activation value A_1"""
        return self.__A1

    @property
    def W2(self):
        """getter for weigths_2 vector"""
        return self.__W2

    @property
    def b2(self):
        """getter for biases_2"""
        return self.__b2

    @property
    def A2(self):
        """getter for activation value A_2"""
        return self.__A2

    def forward_prop(self, X):
        """Forward Propogation for binary class neural net"""
        z1 = np.matmul(self.__W1, X) + self.b1
        self.__A1 = 1/(1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.b2
        self.__A2 = 1/(1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Logistic Regression Cost Function"""
        mth = -1/A.shape[1]
        costs = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return np.sum(costs) * mth

    def evaluate(self, X, Y):
        """Evaluates the predictions made and the cost"""
        A1, predictions = self.forward_prop(X)
        cost = self.cost(Y, predictions)
        eval_bool = predictions >= 0.5
        evaluation = eval_bool.astype(int)
        return evaluation, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Gradient descent method for neural network"""
        mth = 1/A1.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) * mth
        dB2 = mth * np.sum(dZ2, axis=1)
        dsigmoid = A1 * (1 - A1)
        dZ1 = np.matmul(self.__W2.T, dZ2) * dsigmoid
        dW1 = mth * np.matmul(dZ1, X.T)
        dB1 = mth * np.sum(dZ1, axis=1, keepdims=True)
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * dB2)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * dB1)
