#!/usr/bin/env python3
"""
Module contains Class for training agent to play the
cartpole gym env using policy gradient RL method.
"""


import numpy as np
from policy_gradient import policy_gradient


class PolicyGradient():
    """
    Class for training agent to play the
    cartpole gym env using policy gradient RL method.
    """

    def __init__(self, save_res=False):
        """Class constructor."""

        self.gamma = None
        self.lr = None
        self.weights = None
        self.rewards = None
        self.gradients = None
        if save_res:
            from render import Render
            self.render = Render()
        else:
            self.render = False

    def update(self):
        """
        Updates weights towards the policy gradients
        multiplied by the discounted rewards of
        the episode.
        """

        for i in range(len(self.gradients)):
            self.weights = self.weights + \
                (self.lr * self.gradients[i]) * \
                (self.gamma**self.rewards[i:]*self.rewards[i:]).sum()

    def train(self, env, nb_episodes,
              alpha=0.000045, gamma=0.98,
              show_result=False):
        """
        Trains agent to play the cartpole gym env
        using policy gradient RL method.

        Args:
            env: initial environment.
            nb_episodes: number of episodes.
            alpha: the learning rate.
            gamma: the discount factor.

        Return:
            List of total rewards from each episode.
        """

        action_space = env.action_space.n
        state_space = env.observation_space.shape[0]
        tot_rewards = np.empty(nb_episodes)
        self.weights = np.random.rand(state_space, action_space)
        self.lr = alpha
        self.gamma = gamma

        for ep in range(nb_episodes):
            state = env.reset()
            tot_reward, done = 0, False
            self.gradients, self.rewards = [], []

            while not done:

                # With Display present
                # if show_result and ep % 1000 == 0:
                #     env.render()

                if self.render and ep == nb_episodes - 1:
                    self.render.on_step(env)

                action, grad = policy_gradient(state[None, :], self.weights)
                state, reward, done, _ = \
                    env.step(action)

                self.gradients.append(grad)
                self.rewards.append(reward)

                tot_reward += reward

            self.rewards = np.array(self.rewards)
            self.update()

            if ep % 100 == 0:
                print("Episode: {} - Reward: {}".format(ep, tot_reward))
            tot_rewards[ep] = tot_reward

        return tot_rewards
