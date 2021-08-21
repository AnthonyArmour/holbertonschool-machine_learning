#!/usr/bin/env python3
"""Module contains normalization_constants() function"""


import numpy as np


def normalization_constants(X):
    """
       Calculates the normalization (standardization)
       constants of a matrix.

       Args:
         X: numpy.ndarray - matrix to normalize

       Returns:
         The mean and standard deviation
         of each feature, respectively.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
