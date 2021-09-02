#!/usr/bin/env python3
"""
   Module contains
   train_model(network, data, labels, batch_size,
     epochs, verbose=True, shuffle=False):
"""


import tensorflow.keras as k


def train_model(
        network, data, labels, batch_size, epochs,
        verbose=True, shuffle=False):
    """
       Trains a model using mini-batch gradient descent.

       Args:
         network: The model to optimize
         data: numpy.ndarray - input data
         labels: One hot matrix containing Y data
         batch_size: Size of batches for mini
           batch gradient descent
         epochs: Number of epochs to train
         verbose: Boolean determining print status
         shuffle: boolean whether to shuffle data in order

        Returns:
          History object.
    """
    return network.fit(
        x=data, y=labels, batch_size=batch_size,
        epochs=epochs, verbose=verbose,
        shuffle=shuffle
        )
