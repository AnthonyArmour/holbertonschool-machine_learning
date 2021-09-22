#!/usr/bin/env python3
"""
   Module contains
   inception_block(A_prev, filters):
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
      Going Deeper with Convolutions (2014)

    Args:
      A_prev: output from previous layer
      filter: tuple containing filter dims for each layer

    Return:
      Concatenated output of inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal()

    # 1D
    conv1D = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(A_prev)

    # 1D -> 3x3
    conv3_1D = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(A_prev)
    conv3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu', kernel_initializer=init
    )(conv3_1D)

    # 1D -> 5x5
    conv5_1D = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(A_prev)
    conv5 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu', kernel_initializer=init
    )(conv5_1D)

    # 3x3 pool -> 1D
    pool = K.layers.MaxPooling2D(
        (2, 2), strides=(1, 1), padding='same'
    )(A_prev)
    convPool = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(pool)

    out = K.layers.concatenate([conv1D, conv3, conv5, convPool])

    return out
