#!/usr/bin/env python3
"""Forward_prop method using tensorflow"""


import tensorflow as tf


create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward_prop using tensorflow"""
    pred = x
    for i in range(len(layer_sizes)):
        # with tf.variable_scope("layer", reuse=tf.AUTO_REUSE):
        pred = create_layer(pred, layer_sizes[i], activations[i])
        # else:
        #     pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred