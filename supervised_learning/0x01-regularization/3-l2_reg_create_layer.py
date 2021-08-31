#!/usr/bin/env python3
"""
   Module contains
   l2_reg_create_layer(prev, n, activation, lambtha):
"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
       Creates a tensorflow layer that includes L2 regularization.

       Args:
         prev: Tensor containing the output of the previous layer
         n: Number of nodes for new layer
         activation: activation function to be used on layer
         lambtha: L2 regularization parameter

       Returns:
         Output of new layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=init,
        kernel_regularizer=reg
        )(prev)
    return layer
