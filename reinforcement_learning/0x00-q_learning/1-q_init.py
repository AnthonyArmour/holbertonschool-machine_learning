#!/usr/bin/env python3
"""
Module contains function for initializing
q-table.
"""


import numpy as np


def q_init(env):
    """
    Initializes Q-table

    Args:
        env: FrazenLakeEnv

    Return: Q-table as np.ndarray of zeros.
    """

    q = np.zeros((env.desc.size, 4))
    return q
