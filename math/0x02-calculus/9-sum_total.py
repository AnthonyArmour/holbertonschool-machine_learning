#!/usr/bin/env python3
"""summation function"""


def summation_i_squared(n):
    """calculates sigma"""
    sum = 0
    if type(n) is not int:
        return None
    return int((n * (n + 1) * ((2 * n) + 1)) / 6)
