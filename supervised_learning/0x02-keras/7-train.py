#!/usr/bin/env python3
"""
   Module contains
   train_model(network, data, labels, batch_size,
     early_stopping=False, patience=0,
     learning_rate_decay=False, alpha=0.1, decay_rate=1,
     epochs, validation_data=None, verbose=True,
     shuffle=False):
"""


import tensorflow.keras as k


def train_model(
        network, data, labels, batch_size, epochs,
        early_stopping=False, patience=0,
        learning_rate_decay=False, alpha=0.1, decay_rate=1,
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
         learning_rate_decay: Boolean
         alpha: Initial learning rate
         decay_rate: Rate to decay learning rate
         verbose: Boolean determining print status
         shuffle: boolean whether to shuffle data in order

        Returns:
          History object.
    """
    callback = []
    if validation_data and early_stopping is True:
        callback.append(k.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience
            ))

    if validation_data and learning_rate_decay is True:
        def scheduler(epoch):
            """Learning rate scheduler"""
            return alpha / (1+epoch*decay_rate)
        callback.append(k.callbacks.LearningRateScheduler(
            scheduler, verbose=1
        ))

    return network.fit(
        x=data, y=labels, batch_size=batch_size,
        epochs=epochs, verbose=verbose, callbacks=callback,
        validation_data=validation_data, shuffle=shuffle
        )
