#!/usr/bin/env python3
"""
   Modulke contains
   predict(network, data, verbose=False):
"""


import tensorflow.keras as k


def predict(network, data, verbose=False):
    """
       Makes a prediction using a neural network.

       Args:
         network: Model to test
         data: Input data to test model with
         verbose: Boolean

       Returns:
         Prediction for data.
    """
    return network.predict(
        x=data, verbose=verbose
    )
