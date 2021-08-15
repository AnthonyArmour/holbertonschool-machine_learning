#!/usr/bin/env python3
"""create layer function"""


import tensorflow as tf


def create_layer(prev, n, activation):
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.dense(prev, n, activation=activation,
                           kernel_initializer=weights, name="layer")
