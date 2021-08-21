#!/usr/bin/env python3
"""Module contains shuffle_data(X, Y) function"""


import numpy as np


def shuffle_data(X, Y):
    """
       Shuffles the data points in two matrices the same way.

       Args:
         X: numpy.ndarray - (m, nx) matrix to shuffle
         Y: numpy.ndarray - (m, ny) matrix to shuffle

       Returns:
         The shuffled X and Y matrices
    """
