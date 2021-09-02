#!/usr/bin/env python3
"""
   Module contains
   save_weights(network, filename, save_format='h5'):
   load_weights(network, filename):
"""


import tensorflow.keras as k


def save_weights(network, filename, save_format='h5'):
    """
       Saves weights od the model.

       Args:
         network: Model to save
         filename: Filepath
         save_format: Format to save model as

       Return:
         None
    """
    network.save_weights(
        filepath=filename, save_format=save_format
        )
    return None


def load_weights(network, filename):
    """
       Loads weights of a model.

       Args:
         network: model
         filename: Filepath

       Return:
         Loaded model.
    """
    network.load_weights(filepath=filename)
    return None
