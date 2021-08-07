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
        if False in (np.array(layers) > 0):
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
