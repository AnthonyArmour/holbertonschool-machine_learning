#!/usr/bin/env python3
"""
   One hot encode function to be used to reshape
   Y_label vector
"""


import numpy as np


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
    classes = np.amax(labels) + 1
    mat_encode = np.zeros((len(labels), classes))
    for x, label in enumerate(labels):
        mat_encode[x, label] = 1
    return mat_encode
