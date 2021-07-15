#!/bin/usr/env python3
"""array arithmatic"""


def add_arrays(arr1, arr2):
    """adds arrays together"""
    for item in arr1 + arr2:
        try:
            test = len(item)
            return None
        except TypeError:
            continue
    if len(arr1) != len(arr2):
        return None
    sum_array = []
    for x in range(len(arr1)):
        sum_array.append(arr1[x] + arr2[x])
    return sum_array
