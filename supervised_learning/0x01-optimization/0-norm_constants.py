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
         Returns: the mean and standard deviation
         of each feature, respectively.
    """
