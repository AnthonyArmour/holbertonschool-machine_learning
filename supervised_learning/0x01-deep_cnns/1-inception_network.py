#!/usr/bin/env python3
"""
   Module contains
   inception_network():
"""


import tensorflow.keras as K


inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
       Builds the inception network as described in Going
         Deeper with Convolutions (2014).

       Args:
         None

       Return:
         Inception network
    """
    init = K.initializers.he_normal()
    input = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        activation='relu', kernel_initializer=init
        )(input)

    pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
        )(conv1)

    convE = K.layers.Conv2D(
        64, kernel_size=(1, 1), padding='same', activation='relu',
        kernel_initializer=init
    )(pool1)

    conv2 = K.layers.Conv2D(
        192, (3, 3), activation='relu', padding='same',
        kernel_initializer=init
        )(convE)
    pool2 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
        )(conv2)

    inception1 = inception_block(pool2, (64, 96, 128, 16, 32, 32))
    inception2 = inception_block(inception1, (128, 128, 192, 32, 96, 64))

    pool3 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
        )(inception2)

    inception3 = inception_block(pool3, (192, 96, 208, 16, 48, 64))
    inception4 = inception_block(inception3, (160, 112, 224, 24, 64, 64))
    inception5 = inception_block(inception4, (128, 128, 256, 24, 64, 64))
    inception6 = inception_block(inception5, (112, 144, 288, 32, 64, 64))
    inception7 = inception_block(inception6, (256, 160, 320, 32, 128, 128))

    pool4 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
        )(inception7)

    inception8 = inception_block(pool4, (256, 160, 320, 32, 128, 128))
    inception9 = inception_block(inception8, (384, 192, 384, 48, 128, 128))

    pool5 = K.layers.AveragePooling2D(
        (7, 7), strides=(7, 7), padding='same'
        )(inception9)
    dropout = K.layers.Dropout(0.4)(pool5)
    dense = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
        )(dropout)

    return K.Model(inputs=input, outputs=dense)
