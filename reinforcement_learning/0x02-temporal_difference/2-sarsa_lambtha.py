#!/usr/bin/env python3
"""
Module contains the lambda SARSA
reinforcement learning algorithm
using the gym FrozenLake environment.
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


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Args:
        env: openAI environment instance.
        Q: numpy.ndarray of shape (s,a) Q table.
        lambtha: eligibility trace factor.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.
        epsilon: initial threshold for epsilon greedy.
        min_epsilon: minimum value that epsilon should decay to.
        epsilon_decay: decay rate for updating
        epsilon between episodes.

    Return: Q, the updated Q table
    """
    Et = np.zeros_like(Q)
    for ep in range(episodes):
        state = env.reset()
        done = False
        # Et = np.zeros_like(Q)

        action = epsilon_greedy(Q, state, epsilon)

        for step in range(max_steps):
            next_st, reward, done, info = \
                env.step(action)

            a_prime = epsilon_greedy(Q, next_st, epsilon)
            delta = reward + gamma*Q[next_st, a_prime] - Q[state, action]
            Et[state, action] += 1

            Q += alpha * delta * Et
            Et *= lambtha * gamma

            state = next_st
            action = a_prime
            if done:
                break

        epsilon = min_epsilon + (1 - min_epsilon) * \
            np.exp(-epsilon_decay*ep)

    return Q
