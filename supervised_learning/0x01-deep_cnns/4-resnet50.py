#!/usr/bin/env python3
"""
   Module contains
   resnet50():
"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
       Builds the ResNet-50 architecture as described in
         Deep Residual Learning for Image Recognition (2015).

       Args:
         None

       Return:
         Resnet_50
    """
    filters = [
        [64, 64, 256],
        [128, 128, 512],
        [256, 256, 1024],
        [512, 512, 2048]
    ]
    S = [1, 2, 2, 2]
    blocks = [3, 4, 6, 3]

    init = K.initializers.he_normal()
    input = K.Input(shape=(224, 224, 3))

    X = K.layers.ZeroPadding2D((3, 3))(input)
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    for x, flts in enumerate(filters):
        for blcks in range(blocks[x]):
            if blcks == 0:
                X = projection_block(X, flts, S[x])
            else:
                X = identity_block(X, flts)

    pool = K.layers.AveragePooling2D((7, 7), padding='same')(X)

    soft = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
        )(pool)

    return K.Model(inputs=input, outputs=soft)
