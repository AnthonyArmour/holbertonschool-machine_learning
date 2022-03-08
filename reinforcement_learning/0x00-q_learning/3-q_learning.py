#!/usr/bin/env python3
"""
Module contains function for training
a q-learning agent.
"""


import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """
    Performs Q-learning.

    Args:
        env: FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.
        epsilon: initial threshold for epsilon greedy.
        min_epsilon: minimum value that epsilon should decay to.
        epsilon_decay: decay rate for updating epsilon between episodes.

    Return: Q, total_rewards
        Q: Updated Q-table.
        total_rewards: The rewards per episode.
    """

    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_rewards = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_st, reward, done, info = \
                env.step(action)

            Q[state, action] = Q[state, action] * (1 - alpha) + alpha * \
                (reward + gamma * np.max(Q[next_st, :]))

            state = next_st
            ep_rewards += reward

            if done:
                break

        epsilon = min_epsilon + (1 - min_epsilon) * \
            np.exp(-epsilon_decay*ep)
        total_rewards.append(ep_rewards)

    return Q, total_rewards
