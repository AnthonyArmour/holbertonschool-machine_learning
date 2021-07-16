#!/usr/bin/env python3
"""concats matrices on specified axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concats matrices on specified axis"""
    return np.concatenate((mat1, mat2), axis=axis)
