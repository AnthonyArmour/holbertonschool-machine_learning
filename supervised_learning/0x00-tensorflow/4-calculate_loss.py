#!/usr/bin/env python3
"""Calculates Cross Entropy Loss"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates Cross Entropy Loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
