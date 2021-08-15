#!/usr/bin/env python3
"""Calculates accuracy of prediction"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates accuracy of prediction"""
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
