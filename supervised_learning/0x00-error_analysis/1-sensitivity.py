#!/usr/bin/env python3
"""
    Module contains
    sensitivity(confusion)
"""


import numpy as np


def sensitivity(confusion):
    """
        Calculates the sensitivity for each class
          in a confusion matrix.

        Args:
          confusion: numpy.ndarray (classes, classes) - where row
            indices represent the correct labels and column indices
            represent the predicted labels.

        Returns:
          numpy.ndarray (classes,) - containing the sensitivity
            of each class.

    """
    sensitive = np.zeros((confusion.shape[0]))
    for x in range(confusion.shape[0]):
        sensitive[x] = confusion[x][x]/np.sum(confusion[x])
    return sensitive
