#!/usr/bin/env python3
"""
   Module contains
   identity_block(A_prev, filters):
"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
       Builds an identity block as described in Deep Residual
         Learning for Image Recognition (2015):.

       Args:
         A_prev: output of previous layer
         filters: dims of layers for identity block

       Return:
         Activated ouput of identity block
    """
    filts = [(1, 1), (3, 3), (1, 1)]
    init, activation = K.initializers.he_normal(), A_prev

    for x, f in enumerate(filters):
        conv1 = K.layers.Conv2D(
            f, filts[x], padding='same', kernel_initializer=init
            )(activation)
        batch1 = K.layers.BatchNormalization(axis=3)(conv1)
        if x == len(filts) - 1:
            batch1 = K.layers.Add()([batch1, A_prev])
        activation = K.layers.Activation('relu')(batch1)

    return activation
