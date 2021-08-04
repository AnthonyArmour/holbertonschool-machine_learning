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
