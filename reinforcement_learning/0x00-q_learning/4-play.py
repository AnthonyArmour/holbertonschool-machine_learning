#!/usr/bin/env python3
"""
Module contains function for watching
a trained q-learning agent play
FrozenLake.
"""


import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays FrozenLake with trained Q-table agent.

    Args:
        env: FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: Maximum number of steps in the episode.

    Return: Total reward.
    """

    state = env.reset()
    env.render()
    total_reward = 0

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, info = \
            env.step(action)
        total_reward += reward
        env.render()
        if done:
            return total_reward
