#!/usr/bin/env python3
"""
   Module contains
   lenet5(X):
"""


import tensorflow.keras as K


def lenet5(X):
    """
       Modified version of the LeNet-5 architecture using keras.

       Args:
         X: K.Input - (m, 28, 28, 1) input images

       Return:
         tensor for softmax activation
         Adam training operation
         loss tensor
         accuracy tensor
    """
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(
        filters=6, kernel_size=(5, 5), padding='same',
        activation='relu', kernel_initializer=init
    )(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(
        filters=16, kernel_size=(5, 5), padding='valid',
        activation='relu', kernel_initializer=init
    )(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flat = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(
        units=120, kernel_initializer=init, activation='relu'
        )(flat)

    fc2 = K.layers.Dense(
        units=84, kernel_initializer=init, activation='relu'
        )(fc1)

    fc3 = K.layers.Dense(
        units=10, kernel_initializer=init, activation='softmax'
        )(fc2)

    model = K.models.Model(X, fc3)

    model.compile(
        optimizer=K.optimizers.Adam(), loss='categorical_crossentropy',
        metrics=['accuracy']
        )
    return model
