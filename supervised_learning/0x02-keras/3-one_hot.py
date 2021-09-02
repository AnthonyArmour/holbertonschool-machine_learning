#!/usr/bin/env python3
"""
   One hot encode function to be used to reshape
   Y_label vector
"""


import tensorflow.keras as k


def one_hot(labels, classes=None):
    """
       One hot encode function to be used to reshape
         Y_label vector.

       Args:
         labels: Y labels
         classes: number of classes

       Returns:
         One hot matrix.
    """
    return k.utils.to_categorical(labels, classes)
