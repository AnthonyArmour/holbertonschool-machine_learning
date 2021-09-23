#!/usr/bin/env python3
"""
   Module contains
   densenet121(growth_rate=32, compression=1.0):
"""


import tensorflow.keras as K


dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
       Builds the DenseNet-121 architecture as described in
         Densely Connected Convolutional Networks:.

       Args:
         growth_rate: growth rate
         compression: compression factor for transition layer

       Return:
         DenseNet model
    """

    init = K.initializers.he_normal()
    input = K.Input(shape=(224, 224, 3))

    X = K.layers.BatchNormalization(axis=3)(input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        2*growth_rate, (7, 7), padding='same', kernel_initializer=init
        )(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    block, filters = dense_block(X, 64, growth_rate, 6)
    transition, filters = transition_layer(block, filters, compression)

    block, filters = dense_block(transition, filters, growth_rate, 12)
    transition, filters = transition_layer(block, filters, compression)

    block, filters = dense_block(transition, filters, growth_rate, 24)
    transition, filters = transition_layer(block, filters, compression)

    block, _ = dense_block(transition, filters, growth_rate, 16)

    pool = K.layers.AveragePooling2D(
        pool_size=(1, 1), strides=(7, 7), padding='same'
    )(block)

    soft = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
    )(pool)

    return K.Model(inputs=input, outputs=soft)
