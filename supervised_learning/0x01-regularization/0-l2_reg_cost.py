#!/usr/bin/env python3
"""
   Module contains
   l2_reg_cost(cost, lambtha, weights, L, m):
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
       Calculates the cost of a neural network with L2 regularization.

       Args:
         cost: Cost of the network without L2 regularization
         lambtha: The regularization parameter
         weights: Dictionary of the weights and biases
         L: Number of layers in network
         m: Number of data points used
       Returns:
         The cost of the network accounting for L2.
    """
    norm = 0
    for k, w in weights.items():
        if 'W' in k:
            norm += np.sum((w**2).flatten())
    L2 = (lambtha/(2*m)) * norm
    return cost + L2
