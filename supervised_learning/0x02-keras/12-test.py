#!/usr/bin/env python3
"""
   Modulke contains
   test_model(network, data, labels, verbose=True):
"""


import tensorflow.keras as k


def test_model(network, data, labels, verbose=True):
    """
       Tests model on testing data.

       Args:
         network: Model to test
         data: Input data to test model with
         labels: Correct one hot encoded labels
         verbose: Boolean

       Returns:
         Loss and accuracy of model on testing data.
    """
    return network.evaluate(
        x=data, y=labels, verbose=verbose
    )
