#!/usr/bin/env python3
"""
   Module contains a function for
   Principle Component Analysis.
"""


import numpy as np


def pca(X, var=0.95):
    """
       Performs Principle Component Analysis
         on a dataset.

       Args:
        X: numpy.ndarray - (n, d):
            n: number of data points.
            d: number of dimensions in each point.
            - all dimensions have a mean of 0
              across all data points.
        var: fraction of the variance that the PCA
          transformation should maintain.

       Return:
         K principle components that maintain specified
           variance.
    """

    U, vals, eig = np.linalg.svd(X)

    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    eig = (eig.T)[:, idx]

    var_explained = []
    eig_sum = vals.sum()

    for i in range(vals.shape[0]):
        var_explained.append(vals[i]/eig_sum)

    # Cumulative sum
    Csum = np.cumsum(var_explained)

    for i in range(Csum.shape[0]):
        if Csum[i] >= var:
            return eig[:, :i+1]

    return eig
