#!/usr/bin/env python3
"""
   Module contains
   dense_block(X, nb_filters, growth_rate, layers):
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
       Builds a dense block as described in Densely Connected
         Convolutional Networks.

       Args:
         X: output of previous layer
         nb_filters: represents number of filters in X
         growth_rate: groth rate for dense block
         layers: number of layers in dense block

       Return:
         The concatenated output of each layer within the Dense
           Block and the number of filters within the concatenated
           outputs, respectively.
    """

    init = K.initializers.he_normal()
    for i in range(layers):
        layer = K.layers.BatchNormalization(axis=3)(X)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(
            4*growth_rate, (1, 1), padding='same', kernel_initializer=init
        )(layer)
        layer = K.layers.BatchNormalization(axis=3)(layer)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same', kernel_initializer=init
        )(layer)
        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate
    return X, nb_filters
