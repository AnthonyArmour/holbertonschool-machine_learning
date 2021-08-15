#!/usr/bin/env python3
"""Placeholders with tensorflow"""


import numpy as np
import tensorflow as tf
import matplotlib as plt


def create_placeholders(nx, classes):
    """placeholder func"""
    x = tf.placeholder(name="x", dtype=tf.float32, shape=[None, nx])
    y = tf.placeholder(name="y", dtype=tf.float32, shape=[None, classes])
    return x, y
