#!/usr/bin/env python3
"""
   Module contains Priciple component analysis
   function.
"""


import numpy as np


def pca(X, ndim):
    """
       Perform Priciple Component Analysis
         on dataset.

       Args:
        X: numpy.ndarray - shape (n, d)
            n: number of data points.
            d: number of dimensions in each point.
        ndim: new dimensionality of the transformed X.

       Return:
        numpy.ndarray of shape (n, ndim) containing
          transformed version of X.
    """
    X_meaned = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X_meaned)
    W = (V.T)[:, :ndim]

    return X_meaned.dot(W)
