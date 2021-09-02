#!/usr/bin/env python3
"""
   Module contains
   save_model(network, filename):
   load_model(filename):
"""


import tensorflow.keras as k


def save_model(network, filename):
    """
       Saves model.

       Args:
         network: Model to save
         filename: Filepath

       Return:
         None
    """
    k.models.save_model(network, filename)
    return None


def load_model(filename):
    """
       Loads model.

       Args:
         filename: Filepath

       Return:
         Loaded model.
    """
    return k.models.load_model(filename)
