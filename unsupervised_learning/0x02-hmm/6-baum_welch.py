#!/usr/bin/env python3
"""
Module contains function that performs the
Baum-Welch algorithm for finding locally optimal
transition and emission probabilities for a Hidden Markov Model.
"""


import numpy as np
# The forward algorithm for computing the probabilies
# of being in a hidden state gievn all previous observations.
forward = __import__('3-forward').forward
# Backward algo. gives probabilities of being in hidden state,
# given all future observations.
backward = __import__('5-backward').backward


def EM(Observation, Transition, Emission, Initial):
    """
    Expectation Maximization algorithm for updating transition,
    emission, and initial state probabilities to achieve those which
    best generates the observations.

    Args:
        Observations: numpy.ndarray - (T,) Index of the observation.
            T: Number of observations.
        Transition: numpy.ndarray - (M, M) Transition probabilities.
            M: Number of hidden states.
        Emission: numpy.ndarray - (M, N) Emission probabilities.
            N: Number of output states.
        Initial: numpy.ndarray - (M, 1) Starting probabilities.

    Return:
        Emission, Transition, Initial after one update step.
    """

    T = Observation.size
    M, N = Emission.shape
    _, F = forward(Observation, Emission, Transition, Initial)
    _, B = backward(Observation, Emission, Transition, Initial)

    # F[i, j] is the probability of being in hidden state i at time j given
    # the previous observations.

    # B[i, j] is the probability of generating the future observations from
    # hidden state i at time j.

    Xi = np.zeros((T-1, M, M))

    for t in range(T-1):
        op = F[:, t].reshape(M, 1) * Transition * Emission[:, Observation[t+1]]
        op = op * B[:, t+1]
        Xi[t, :, :] = op.copy()

    Xi = Xi / Xi.sum(axis=(1, 2)).reshape(T-1, 1, 1)

    Transition = Xi.sum(axis=0) / Xi.sum(axis=(0, 2)).reshape(M, 1)

    for k in range(N):
        idxs = Observation[:T-1] == k
        Emission[:, k] = Xi[idxs, :, :].sum(axis=(0, 1))/Xi.sum(axis=(0, 1))

    Initial = Xi[0].sum(axis=1)

    return Transition, Emission, Initial.reshape(M, 1)


def baum_welch(Observation, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for finding locally optimal
    transition and emission probabilities for a Hidden Markov Model.

    Args:
        Observations: numpy.ndarray - (T,) Index of the observation.
            T: Number of observations.
        Transition: numpy.ndarray - (M, M) Initialized transition
        probabilities.
            M: Number of hidden states.
        Emission: numpy.ndarray - (M, N) Initialized emission probabilities.
            N: Number of output states.
        Initial: numpy.ndarray - (M, 1) Initialized starting probabilities.
        iterations: Number of times expectation-maximization should
        be performed.

    Return:
        Converged Transition, Emission, or None, None on failure.
    """

    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if Observation.shape[0] == 0:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    for i in range(iterations):
        Transition, Emission, Initial = EM(
            Observation, Transition, Emission, Initial)

    return Transition, Emission
