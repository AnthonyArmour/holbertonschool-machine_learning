#!/usr/bin/env python3
"""Deep Neural Network Module"""


import numpy as np


class DeepNeuralNetwork():
    """
       Deep Neural Network Class, used for binary
       classification on handwritten digits
    """

    def __init__(self, nx, layers):
        """init method for DeepNeuralNetwork class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any([True for n in layers if n <= 0]) is True:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layers = [nx] + layers
        for x, l in enumerate(layers[1:], start=1):
            self.__weights["W{}".format(x)] = (
                np.random.randn(l, layers[x - 1]) *
                np.sqrt(2/(layers[x - 1]))
            )
            self.__weights["b{}".format(x)] = np.zeros((l, 1))

    @property
    def cache(self):
        """getter for cache dictionary"""
        return self.__cache

    @property
    def L(self):
        """getter for num of layers, L"""
        return self.__L

    @property
    def weights(self):
        """getter for weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
           Forward Propagation method for
           Deep Neural Network using sigmoid
           activation function
        """
        self.__cache["A0"] = X
        for layer in range(1, self.__L + 1):
            Z = (
                np.matmul(self.__weights["W{}".format(layer)],
                          self.__cache["A{}".format(layer - 1)]) +
                self.__weights["b{}".format(layer)]
                )
            self.__cache["A{}".format(layer)] = 1/(1 + np.exp(-Z))
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Logistic Regression Cost Function"""
        mth = -1/A.shape[1]
        costs = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return np.sum(costs) * mth

    def evaluate(self, X, Y):
        """Evaluates the predictions made and the cost"""
        predictions, cache = self.forward_prop(X)
        cost = self.cost(Y, predictions)
        eval_bool = predictions >= 0.5
        evaluation = eval_bool.astype(int)
        return evaluation, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        mth = 1/cache["A1"].shape[1]
        partials = {}
        new_weights = {}
        partials["Z{}".format(self.__L)] = cache["A{}".format(self.__L)] - Y
        partials["W{}".format(self.__L)] = (
            mth * np.matmul(partials["Z{}".format(self.__L)],
                            cache["A{}".format(self.__L - 1)].T)
            )
        new_weights["W{}".format(self.__L)] = (
            self.__weights["W{}".format(self.__L)] -
            (alpha * partials["W{}".format(self.__L)])
        )
        partials["b{}".format(self.__L)] = (
            mth * np.sum(partials["Z{}".format(self.__L)],
                         axis=1, keepdims=True)
        )
        new_weights["b{}".format(self.__L)] = (
            self.__weights["b{}".format(self.__L)] -
            (alpha * partials["b{}".format(self.__L)])
        )

        for layer in range(self.__L - 1, 0, -1):
            partials["Z{}".format(layer)] = (
                np.matmul(self.__weights["W{}".format(layer + 1)].T,
                          partials["Z{}".format(layer + 1)]) *
                (cache["A{}".format(layer)] * (1 - cache["A{}".format(layer)]))
            )
            partials["W{}".format(layer)] = (
                mth * np.matmul(partials["Z{}".format(layer)],
                                cache["A{}".format(layer - 1)].T)
            )
            new_weights["W{}".format(layer)] = (
                self.__weights["W{}".format(layer)] -
                (alpha * partials["W{}".format(layer)])
            )
            partials["b{}".format(layer)] = (
                mth * np.sum(partials["Z{}".format(layer)],
                             axis=1, keepdims=True)
            )
            new_weights["b{}".format(layer)] = (
                self.__weights["b{}".format(layer)] -
                (alpha * partials["b{}".format(layer)])
            )
        self.__weights = new_weights
