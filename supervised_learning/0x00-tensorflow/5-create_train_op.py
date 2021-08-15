#!/usr/bin/env python3
"""Computes gradient descent"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """Computes gradient descent"""
    optimize = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    return optimize.minimize(loss=loss)
