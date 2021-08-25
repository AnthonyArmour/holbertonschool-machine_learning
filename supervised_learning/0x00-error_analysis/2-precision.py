#!/usr/bin/env python3
"""
    Module contains
    precision(confusion):
"""


import numpy as np


def precision(confusion):
    """
        Calculates the precision for each class in
          a confusion matrix.

        Args:
          confusion: numpy.ndarray (classes, classes) - where row
            indices represent the correct labels and column indices
            represent the predicted labels.

        Returns:
          numpy.ndarray (classes,) - containing the precision
            of each class.

    """
    precision = np.zeros((confusion.shape[0]))
    for x in range(confusion.shape[0]):
        precision[x] = confusion[x][x]/np.sum(confusion[:, x])
    return precision
