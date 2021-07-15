#!/usr/bin/env python3
"""adds two matrices"""


def add_matrices2D(mat1, mat2):
    """adds 2d matrices"""
    if len(mat1) != len(mat2):
        return None
    sum_matrix = [[] for x in range(len(mat1))]
    for x, row in enumerate(mat1):
        for y, item in enumerate(mat2[x]):
            sum_matrix[x].append(item + row[y])
    return sum_matrix
