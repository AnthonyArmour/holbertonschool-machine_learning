#!/usr/bin/env python3
"""
Module contains the lambda TD
reinforcement learning algorithm
using the gym FrozenLake environment.
"""


import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Args:
        env: openAI environment instance.
        V: numpy.ndarray of shape (s,) value estimate.
        policy: function that takes in a state and returns
        the next action to take.
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.

    Return: V, updated value estimate
    """

    for ep in range(episodes):
        state = env.reset()
        Et = np.zeros(V.shape[0])
        for step in range(max_steps):
            action = policy(state)
            nx_state, reward, done, _ = \
                env.step(action)

            Et *= gamma * lambtha
            Et[state] += 1

            tde = reward + (gamma * V[nx_state]) - V[state]

            V += alpha * tde * Et

            if done:
                break

            state = nx_state

    return V
