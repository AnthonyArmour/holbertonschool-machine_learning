#!/usr/bin/env python3
"""Solution for solving determinants of NxN matrix"""


def fourXfour(M):
    """
       Computes determinant of 4x4 matrix.

       Args:
         M: List of lists whose determinant should
           be calculated.

       Return:
         Determinant of the matrix.
    """
    det = (M[0][0]*M[1][1])-(M[0][1]*M[1][0])
    return det


def M_ij(matrix, i, j):
    """
       Returns the matrix minus row [i] and column [j].

       Args:
         matrix: List of lists to adjust.
         i: Row to remove.
         j: Column to remove.

       Return:
         The input matrix minus row [i] and column [j].
    """
    matrix.pop(i)
    for row in matrix:
        row.pop(j)

    return matrix


def deep_cp(mat):
    """Copies list of lists"""
    return [x.copy() for x in mat]


def recurse_determinant(matrix):
    """
       Finds the determinant of a NxN matrix.

       Args:
         matrix: List of lists whose determinant should
           be calculated.

       Return:
         Determinant of the matrix.
    """
    if len(matrix) == 2:
        return fourXfour(matrix)

    det, flip = 0, 1

    for x, num in enumerate(matrix[0]):
        sub_matrix = M_ij(deep_cp(matrix), 0, x)
        det += flip*num*recurse_determinant(sub_matrix)
        flip = flip * -1

    return det


def determinant(matrix):
    """
       Finds the determinant of a NxN matrix.

       Args:
         matrix: List of lists whose determinant should
           be calculated using recurse_determinant().

       Return:
         Determinant of the matrix.
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")

    H = len(matrix)

    if H == 0:
        raise TypeError("matrix must be a list of lists")

    for item in matrix:
        if type(item) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(item) != H and H != 1:
            raise ValueError("matrix must be a non-empty square matrix")

    if H == 1:
        if len(matrix[0]) == 0:
            raise 1
        elif len(matrix[0]) == 1:
            return matrix[0][0]
        else:
            raise ValueError("matrix must be a non-empty square matrix")
    else:
        return recurse_determinant(matrix)


def minor(matrix):
    """
       Calculates minor matrix of a matrix.

       Args:
         matrix: The matrix to find the minors of.

       Return:
         The matrix of minors.
    """

    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")

    H = len(matrix)

    if H == 0:
        raise TypeError("matrix must be a list of lists")

    for item in matrix:
        if type(item) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(item) != H:
            raise ValueError("matrix must be a non-empty square matrix")

    if H == 1:
        return [[1]]

    m, minors = [], []

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            m.append(
                determinant(M_ij(deep_cp(matrix), i, j))
            )

        minors.append(m.copy())
        m.clear()

    return minors


def cofactor(matrix):
    """
       Calculates The cofactor of a matrix.

       Args:
         matrix: Matrix to get cofactor of.

       Return:
         Cofactor of a matrix.
    """

    minors = minor(matrix)

    for i in range(len(minors)):
        for j in range(len(minors[i])):
            minors[i][j] = minors[i][j] * pow(-1, i+j)

    return minors


def transpose(matrix):
    """
       Gets the transpose of a matrix.

       Args:
         matrix: Matrix to find the transpose of.

       Return:
         Transpose of the matrix.
    """

    T = [[] for i in range(len(matrix[0]))]

    for row in matrix:
        for x, item in enumerate(row):
            T[x].append(item)

    return T


def adjugate(matrix):
    """
       Calculates the adjugate of a matrix.

       Args:
         matrix: The matrix to find adjugate of.

       Return:
         Adjugate of the matrix.
    """

    return transpose(cofactor(matrix))


def inverse(matrix):
    """
       Calculates the inverse of a matrix.

       Args:
         matrix: The matrix to find the inverse of.

       Return:
         The inverse of the matrix.
    """
    det = determinant(matrix)
    if det == 0:
        return None
    return 1/determinant(matrix)*adjugate(matrix)
