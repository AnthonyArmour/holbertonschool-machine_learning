#!/usr/bin/env python3

def matrix_transpose(matrix):
    """transposes a 2d matrix"""
    mTransposed = [[] for x in range(len(matrix[0]))]
    for x, arr in enumerate(mTransposed):
        for row in matrix.copy():
            arr.append(row[x])
    return mTransposed
