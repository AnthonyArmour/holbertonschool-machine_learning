#!/usr/bin/env python3
"""
   Module contains
   dropout_create_layer(prev, n, activation, keep_prob):
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
       Creates a layer of a neural network using dropout.

       Args:
         prev: Tensor containing the output of the previous layer
         n: Number of nodes for new layer
         activation: activation function to be used on layer
         keep_prob: probability of keeping nodes in layer

       Returns:
         Output of new layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.layers.Dropout(rate=(1-keep_prob))
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=init,
        kernel_regularizer=reg
        )(prev)
    return layer
