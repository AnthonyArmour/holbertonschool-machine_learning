#!/usr/bin/env python3
"""finds dimensions of matrix"""


def matrix_shape(matrix):
    """finds dimensions of a matrix"""
    mat = matrix
    dim = []
    while True:
        try:
            dim.append(len(mat))
            mat = mat[0]
        except TypeError:
            break
    return dim
