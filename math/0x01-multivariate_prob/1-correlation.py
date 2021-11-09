#!/usr/bin/env python3
"""
   Module contains function for
   computing correlation matrix.
"""


import numpy as np
from numpy.core.numeric import outer


def correlation(C):
    """
       Calculates a correlation matrix.

       Args:
        C: numpy.ndarray - Covariant matrix.

       Return:
        numpy.ndarray - Correlation matrix.
    """

    Di = np.sqrt(np.diag(C))
    outer_Di = np.outer(Di, Di)
    corr = C / outer_Di
    corr[C == 0] = 0
    return corr
