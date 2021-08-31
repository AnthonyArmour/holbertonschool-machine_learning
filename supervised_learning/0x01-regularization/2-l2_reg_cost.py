#!/usr/bin/env python3
"""
   Module contains
   l2_reg_cost(cost):
"""


import tensorflow as tf


def l2_reg_cost(cost):
    """
       Calculates the cost of a neural network with L2 regularization.

       Args:
         cost: tensor containing the cost of the network
           without L2 regularization

       Returns:
         Tensor containing the cost of the network accounting for
           L2 regularization.
    """
    return cost + tf.losses.get_regularization_losses()
