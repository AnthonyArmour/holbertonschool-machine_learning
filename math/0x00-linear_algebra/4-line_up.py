#!/usr/bin/env python3
"""array arithmatic"""


def add_arrays(arr1, arr2):
    """adds arrays together"""
    if len(arr1) != len(arr2):
        return None
    sum_array = []
    for x in range(len(arr1)):
        sum_array.append(arr1[x] + arr2[x])
    return sum_array
