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
    mat_encode = np.zeros((len(Y), classes))
    for x, label in enumerate(Y):
        mat_encode[x, label] = 1
    return mat_encode.T
