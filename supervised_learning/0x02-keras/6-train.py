#!/usr/bin/env python3
"""
   Module contains
   train_model(network, data, labels, batch_size,
     early_stopping=False, patience=0,
     epochs, validation_data=None, verbose=True,
     shuffle=False):
"""


import tensorflow.keras as k


def train_model(
        network, data, labels, batch_size, epochs,
        early_stopping=False, patience=0,
        validation_data=None, verbose=True, shuffle=False):
    """
       Trains a model using mini-batch gradient descent.

       Args:
         network: The model to optimize
         data: numpy.ndarray - input data
         labels: One hot matrix containing Y data
         batch_size: Size of batches for mini
           batch gradient descent
         early_stopping: Boolean
         patience: Patince value for early stopping
         epochs: Number of epochs to train
         verbose: Boolean determining print status
         shuffle: boolean whether to shuffle data in order

        Returns:
          History object.
    """
    if validation_data is not None and early_stopping is True:
        callback = [k.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience
            )]
    else:
        callback = None

    return network.fit(
        x=data, y=labels, batch_size=batch_size,
        epochs=epochs, verbose=verbose, callbacks=callback,
        validation_data=validation_data, shuffle=shuffle
        )
