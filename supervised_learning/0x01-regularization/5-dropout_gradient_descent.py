#!/usr/bin/env python3
"""
   Module contains
   dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
       Updates the weights of a neural network with Dropout
         regularization using gradient descent.

       Args:
         alpha: Learning rate
         lambtha: The regularization parameter
         weights: Dictionary of the weights and biases
         L: Number of layers in network
         cache: Dictionary of the outputs and dropout masks of each
           layer of the neural network.
         Y: mumpy.ndarray - Contains one hot encoded labels
       Returns:
         The weights and biases of the network should be
           updated in place.
    """
    dZL, m = cache["A{}".format(L)] - Y, Y.shape[1]
    for x in range(L, 0, -1):
        dw = (np.matmul(dZL, cache["A{}".format(x-1)].T)/m)
        db = np.sum(dZL, axis=1, keepdims=True)/m

        weights["b{}".format(x)] -= (alpha * db)
        if x != 1:
            mask = cache["D{}".format(x-1)]
            dz = (1 - (cache["A{}".format(x-1)]**2))
            dZL = np.matmul(weights["W{}".format(x)].T, dZL)*dz*mask
            dZL /= keep_prob

        weights["W{}".format(x)] -= alpha * dw
