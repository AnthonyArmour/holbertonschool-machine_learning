#!/usr/bin/env python3
"""
   Module contains
   save_config(network, filename):
   load_config(filename):
"""


import tensorflow.keras as k


def save_config(network, filename):
    """
       Saves model as json.

       Args:
         network: Model to save
         filename: Filepath

       Return:
         None
    """
    model = network.to_json()
    with open(filename, "w") as fp:
        fp.write(model)
    return None


def load_config(filename):
    """
       Loads model from json.

       Args:
         filename: Filepath

       Return:
         Loaded model.
    """
    with open(filename, "r") as fp:
        model = fp.read()
    return k.models.model_from_json(model)
