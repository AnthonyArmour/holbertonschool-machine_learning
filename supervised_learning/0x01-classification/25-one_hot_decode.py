#!/usr/bin/env python3
"""decodes a one_hot_encoded matrix"""


import numpy as np


def one_hot_decode(one_hot):
    """decodes a one_hot_encoded matrix"""
    decode = []
    for x in range(one_hot.shape[1]):
        decode = np.append(decode, np.where(one_hot.T[x] == 1)[0][0])
    return np.array(decode).astype(int)
