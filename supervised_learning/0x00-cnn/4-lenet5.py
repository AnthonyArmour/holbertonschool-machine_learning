#!/usr/bin/env python3
"""
   Module contains
   lenet5(x, y):
"""


import tensorflow as tf


def lenet5(x, y):
    """
       Modified version of the LeNet-5 architecture using tensorflow.

       Args:
         x: tf.placeholder - (m, 28, 28, 1) input images
         y: tf.placeholder - (m, 10) one-hot labels

       Return:
         tensor for softmax activation
         Adam training operation
         loss tensor
         accuracy tensor
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(
        filters=6, kernel_size=(5, 5), padding='same',
        activation='relu', kernel_initializer=init
    )(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(
        filters=16, kernel_size=(5, 5), padding='valid',
        activation='relu', kernel_initializer=init
    )(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flat = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(
        units=120, kernel_initializer=init, activation='relu'
        )(flat)

    fc2 = tf.layers.Dense(
        units=84, kernel_initializer=init, activation='relu'
        )(fc1)

    fc3 = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)
    prediction = fc3

    loss = tf.losses.softmax_cross_entropy(y, fc3)
    Adam = tf.train.AdamOptimizer().minimize(loss)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    softmax = tf.nn.softmax(prediction)
    return softmax, Adam, loss, accuracy
