#!/usr/bin/env python3
"""
   Module contains
   optimize_model(network, alpha, beta1, beta2):
"""


import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    """
       Sets up Adam optimization for a keras model with
         categorical crossentropy loss and accuracy metrics.

       Args:
         network: The model to optimize
         alpha: Learning rate
         beta1: First adam opt parameter
         beta2: Second adam opt parameter

        Returns:
          None.
    """
    Adam = k.optimizers.Adam
    network.compile(
        optimizer=Adam(alpha, beta1, beta2),
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )
