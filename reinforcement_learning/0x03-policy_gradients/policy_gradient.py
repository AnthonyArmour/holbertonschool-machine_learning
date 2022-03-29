#!/usr/bin/env python3
"""
Module contains functions for
computing policy gradients.
"""


import numpy as np


def policy(state, weight):
    """
    Computes policy with a weight of a matrix.

    Args:
        state: State feature vector.
        weight: Matrix of random weights.

    Return:
        Softmax distribution of the given weight
        and state.
    """
    p = state.dot(weight)
    exp = np.exp(p)
    return exp/np.sum(exp)


def policy_gradient(state, weight):
    """
    Computes policy with a weight of a matrix.

    Args:
        state: State feature vector.
        weight: Matrix of random weights.

    Return: action, gradient
    """
    actions = policy(state, weight)
    action = np.random.choice(actions.size, p=actions[0])

    grad = np.diagflat(actions.T) - actions.T.dot(actions)
    grad = grad[action, :]
    log = grad / actions[0, action]
    gradient = state.T.dot(log[None, :])
    return action, gradient
