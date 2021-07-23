#!/usr/bin/env python3
"""summation function"""


def summation_i_squared(n):
    """calculates sigma"""
    sum = 0
    for i in range(1, n + 1):
        sum += i * i
    return sum
