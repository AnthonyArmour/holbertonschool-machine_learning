#!/usr/bin/env python3
"""concats 2d matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """cancats 2D matrices"""
    if axis > 1 or axis < 0:
        return None
    matrix = []
    for row in mat1:
        matrix.append(row.copy())
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat2:
            matrix.append(row.copy())
        return matrix
    elif len(mat1) != len(mat2):
        return None
    for x, row in enumerate(matrix):
        for item in mat2[x].copy():
            row.append(item)
    return matrix
