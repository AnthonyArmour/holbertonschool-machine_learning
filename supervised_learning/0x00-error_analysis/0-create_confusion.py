#!/usr/bin/env python3
"""
    Module contains
    create_confusion_matrix(labels, logits)
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
        Creates a confusion matrix.

        Args:
          labels: numpy.ndarray - contains correct
            labels for each data point.
          logits: numpy.ndarray - contains predicted labels

        Returns:
          Confusion matrix - numpy.ndarray
    """

    K = len(labels[0])
    result = np.zeros((K, K))
    labelsn = np.where(labels == 1)[1]
    logitsn = np.where(logits == 1)[1]
    for i in range(len(labelsn)):
        result[labelsn[i]][logitsn[i]] += 1
    return result
