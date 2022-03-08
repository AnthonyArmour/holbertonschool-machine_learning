#!/usr/bin/env python3
"""
Module contains function for
performing epsilon greedy.
"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epislon greddy function to determine next action.

    Args:
        Q: numpy.ndarray containing the q-table.
        state: Current state.
        epsilon: Epsilon to use for the calculation.

    Return: Next action index.
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(0, Q.shape[1])
    else:
        return np.argmax(Q[state, :])
