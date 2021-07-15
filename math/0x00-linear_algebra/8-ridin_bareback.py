#!/usr/bin/env python3
"""Matrix multiplication"""


def mat_mul(mat1, mat2):
    """matrix mutltiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    matrix = [[] for x in range(len(mat1))]
    x = mat2
    mat2T = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]

    eSum = 0
    for x, row in enumerate(mat1):
        for xx, row in enumerate(mat2T):
            for y, element in enumerate(row):
                eSum += mat1[x][y] * mat2T[xx][y]
            if type(eSum) is int:
                matrix[x].append(int(eSum))
            else:
                matrix[x].append(float(eSum))
            eSum = 0
    return matrix
