#!/usr/bin/env python3
"""
   Module contains
   projection_block(A_prev, filters, s=2):
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
       Builds a projection block as described in Deep Residual
         Learning for Image Recognition (2015).

       Args:
         A_prev: output of previous layer
         filters: dims of layers for projection block

       Return:
         Activated ouput of projection block
    """
    filts = [(1, 1), (3, 3), (1, 1)]
    init, activation = K.initializers.he_normal(), A_prev

    for x, f in enumerate(filters):
        if x == 0:
            stride = (s, s)
        else:
            stride = (1, 1)
        conv1 = K.layers.Conv2D(
            f, filts[x], strides=stride, padding='same',
            kernel_initializer=init
            )(activation)
        batch1 = K.layers.BatchNormalization(axis=3)(conv1)
        if x < len(filts) - 1:
            activation = K.layers.Activation('relu')(batch1)

    projection = K.layers.Conv2D(
        filters[-1], (1, 1), strides=(s, s), padding='same',
        kernel_initializer=init
    )(A_prev)
    batch_projection = K.layers.BatchNormalization(axis=3)(projection)

    out = K.layers.Add()([batch1, batch_projection])
    activated_out = K.layers.Activation('relu')(out)

    return activated_out
