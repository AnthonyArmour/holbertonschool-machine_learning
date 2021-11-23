#!/usr/bin/env python3
"""
Module contains function for computing
total intra-cluster variance given dataset X
and cluster matrix C.
"""


import numpy as np


def Kassignments(X, clusters):
    """
    Returns list of indexes corresponding to
    min euclidean distance of each point in X to
    each point in clusters.
    """
    points, dims = X.shape
    x = X.reshape(points, 1, dims)
    x = x - np.repeat(clusters[np.newaxis, ...], points, axis=0)
    dist = np.linalg.norm(x, axis=2)

    mins = np.argmin(dist, axis=1)
    return mins


def variance(X, C):
    """
    Calculates the total intra-cluster variance
      for a dataset.

    Args:
        X: numpy.ndarray - (n, d) containing the data set.
        C: numpy.ndarray - (k, d) containing the centroid
          means for each cluster.

    Return:
        Variance, or None on failure.
    """

    assignments = Kassignments(X, C)
    Karanged = C[assignments]

    return ((X-Karanged)**2).sum()
