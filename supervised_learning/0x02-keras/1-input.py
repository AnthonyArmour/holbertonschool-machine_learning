#!/usr/bin/env python3
"""
   Module contains
   build_model(nx, layers, activations, lambtha, keep_prob):
"""


import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
       Builds a neural network with the Keras library.

       Args:
         nx: Number of input features to the network
         layers: List containing number of nodes for each layer
         activations: List containing activation functions for
           each layer
         lambtha: The L2 regularization parameter
         keep_prob: Probability that a node will be kept
           from dropout

        Returns:
          Keras model.
    """
    inputs = k.layers.Input(shape=(nx,))
    for x, i in enumerate(layers):
        if x == 0:
            layer = k.layers.Dense(
                        i, activation=activations[x],
                        kernel_regularizer=k.regularizers.l2(lambtha)
                )(inputs)
        else:
            layer = k.layers.Dense(
                i, activation=activations[x],
                kernel_regularizer=k.regularizers.l2(lambtha)
                )(layer)
        if x < len(layers) - 1:
            layer = k.layers.Dropout(1-keep_prob)(layer)
    model = k.Model(inputs=inputs, outputs=layer)
    return model
