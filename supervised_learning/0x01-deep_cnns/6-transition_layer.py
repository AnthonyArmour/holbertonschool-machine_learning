#!/usr/bin/env python3
"""
   Module contains
   transition_layer(X, nb_filters, compression):
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
       Builds a transition layer as described in Densely
         Connected Convolutional Networks.

       Args:
         X: output of previous layer
         nb_filters: represents number of filters in X
         compression: compression factor for transition layer

       Return:
         The output of the transition layer and the number of
           filters within the output, respectively.
    """

    init = K.initializers.he_normal()
    flts = int(compression*nb_filters)

    layer = K.layers.BatchNormalization(axis=3)(X)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(
        flts, (1, 1), padding='same', kernel_initializer=init
        )(layer)
    pooled = K.layers.AveragePooling2D((2, 2))(layer)

    return pooled, flts
