#!/usr/bin/env python3
"""Module contains normalize(X, m, s) function"""


import numpy as np


def normalize(X, m, s):
    """
       Normalizes (standardizes) a matrix.

       Args:
         X: numpy.ndarray - matrix to normalize
         m: numpy.ndarray -  that contains the mean of all features of X
         s: numpy.ndarray - contains the standard deviation of
            all features of X

       Returns:
         The normalized X matrix.
    """
    return (X - m)/s
