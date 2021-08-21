#!/usr/bin/env python3
"""
   Module contains
   create_batch_norm_layer(prev, n, activation):
"""


import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
       Creates a batch normalization layer for a neural network in tensorflow.

       Args:
         prev is the activated output of the previous layer
         n: number of nodes in the layer to be created
         activation: activation function that should be used on the output
           of the layer

       Returns:
         A tensor of the activated output for the layer.
    """
