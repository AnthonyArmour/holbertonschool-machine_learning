#!/usr/bin/env python3
"""
Module contains the monte carlo
reinforcement learning algorithm
using the gym FrozenLake environment.
"""


import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Args:
        env: openAI environment instance.
        V: numpy.ndarray of shape (s,) value estimate.
        policy: function that takes in a state and returns
        the next action to take.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.

    Return: V, updated value estimate
    """

    for ep in range(episodes):
        state = env.reset()
        sars = []
        for step in range(max_steps):
            action = policy(state)
            nx_state, reward, done, _ = \
                env.step(action)
            sars.append((state, reward))
            if done:
                break
            state = nx_state

        sars = np.array(sars, dtype=int)

        G = 0

        for i, sar in enumerate(sars[::-1]):
            G = (gamma * G) + sar[1]

            if sar[0] not in sars[:ep, 0]:
                V[sar[0]] = V[sar[0]] + (alpha * (G - V[sar[0]]))

    return V
