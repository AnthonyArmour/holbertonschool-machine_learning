#!/usr/bin/env python3
"""
   One hot encode function to be used to reshape
   Y_label vector
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
       One hot encode function to be used to reshape
       Y_label vector
    """
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    if classes < 2 or classes < np.amax(Y):
        return None
    mat_encode = np.zeros((len(Y), classes))
    for x, label in enumerate(Y):
        mat_encode[x, label] = 1
    return mat_encode.T
