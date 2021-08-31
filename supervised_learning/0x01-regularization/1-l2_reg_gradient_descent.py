#!/usr/bin/env python3
"""
   Module contains
   l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
       Updates the weights and biases of a neural network using gradient
         descent with L2 regularization.

       Args:
         alpha: Learning rate
         lambtha: The regularization parameter
         weights: Dictionary of the weights and biases
         L: Number of layers in network
         cache: Dictionary of the outputs of each layer of the
           neural network.
         Y: mumpy.ndarray - Contains one hot encoded labels
       Returns:
         The weights and biases of the network should be
           updated in place.
    """
    dZL, m = cache["A{}".format(L)] - Y, Y.shape[1]
    for x in range(L, 0, -1):
        l2w = ((alpha*lambtha)/m)
        dw = alpha * (np.matmul(dZL, cache["A{}".format(x-1)].T))
        weights["b{}".format(x)] = np.sum(dZL, axis=1, keepdims=True)/m
        if x != 1:
            dz = cache["A{}".format(x-1)] * (1 - cache["A{}".format(x-1)])
            dZL = np.matmul(weights["W{}".format(x)].T, dZL) * dz
        weights["W{}".format(x)] -= l2w - dw
