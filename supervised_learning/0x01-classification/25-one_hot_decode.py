#!/usr/bin/env python3
"""decodes a one_hot_encoded matrix"""


import numpy as np


def one_hot_decode(one_hot):
    """decodes a one_hot_encoded matrix"""
    if type(one_hot) is not np.ndarray:
        return None
    if one_hot.ndim != 2:
        return None
    return np.argmax(one_hot.T, axis=1)
