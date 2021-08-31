#!/usr/bin/env python3
"""
   Module contains
   dropout_forward_prop(X, weights, L, keep_prob):
"""


import numpy as np


def activate(layer, W, A, softmax=False):
    """
       Tanh method for hidden layers.

       Args:
         layer: layer to activate
           W: weights and biases dictionary
           A: Previous activations dictionary
           softmax: True if using softmax activation
       Returns:
          Activation for layer.
    """
    Z = (
        np.matmul(W["W{}".format(layer)],
                  A["A{}".format(layer - 1)]) +
        W["b{}".format(layer)]
        )
    if softmax is False:
        return (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    else:
        T = np.exp(Z)
        return T/np.sum(T, axis=0)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
       Conducts forward propagation using Dropout.

       Args:
         X: numpy.ndarray - input data (nx, m)
         weights: Dictionary containing weights and biases
         L: Number of layers
         keep_prob: probability that a node will be kept

       Returns:
         Dictionary containing activations and dropout masks.
    """
    A = {"A0": X}
    for i in range(1, L+1):
        if i != L:
            a = activate(i, weights, A, softmax=False)
            mask = (
                np.random.rand(a.shape[0], a.shape[1]) < keep_prob
                ).astype(int)
            A["A{}".format(i)] = (a*mask)/keep_prob
            A["D{}".format(i)] = mask
        else:
            A["A{}".format(i)] = activate(i, weights, A, softmax=True)
    return A
