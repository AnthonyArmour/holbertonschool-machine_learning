#!/usr/bin/env python3
"""Matrix multiplication"""


def matrix_transpose(matrix):
    """transposes a 2d matrix"""
    mTransposed = [[] for x in range(len(matrix[0]))]
    for x, arr in enumerate(mTransposed):
        for row in matrix.copy():
            arr.append(row[x])
    return mTransposed


def mat_mul(mat1, mat2):
    """matrix mutltiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    matrix = [[] for x in range(len(mat1))]
    mat2_transposed = matrix_transpose(mat2)

    eSum = 0
    for x, row in enumerate(mat1):
        for xx, row in enumerate(mat2_transposed):
            for y, element in enumerate(row):
                eSum += mat1[x][y] * mat2_transposed[xx][y]
            if type(eSum) is int:
                matrix[x].append(int(eSum))
            else:
                matrix[x].append(float(eSum))
            eSum = 0
    return matrix
