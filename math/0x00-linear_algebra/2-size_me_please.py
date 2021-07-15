#!/usr/bin/env python3

def matrix_shape(matrix):
    """finds dimensions of a matrix"""
    mat = matrix
    dim = []
    while True:
        try:
            dim.append(len(mat))
            mat = mat[0]
        except:
            break
    return dim

