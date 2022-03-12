#!/usr/bin/env python3
"""
Module contains class, methods, and script
for creating and training a deep Q agent
to play Atari breakout.
"""


from __future__ import division
import numpy as np
import gym
from gym import wrappers
import os.path
import pickle
from MyDQNAgent import MyDQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam
import keras.backend as backend
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import matplotlib.animation as animation
from IPython.display import HTML


ENV_NAME = 'BreakoutDeterministic-v0'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class Process(Processor):
    """Processor class"""

    def process_observation(self, obs):
        """
        Preprocess observation.

        Args:
            obs: Observation.
        """
        assert obs.ndim == 3
        img = Image.fromarray(obs)
        img = img.resize(INPUT_SHAPE).convert('L')
        img = np.array(img)
        assert img.shape == INPUT_SHAPE
        return img.astype('uint8')

    def process_state_batch(self, batch):
        """
        Process state.

        Args:
            batch: Batch of states.
        """
        processed = batch.astype('float32') / 255
        return processed

    def process_reward(self, reward):
        """
        Process reward.

        Args:
            reward: reward.
        """
        return np.clip(reward, -1., 1.)


def create_model(nb_actions):
    """
    Creates a convalutional neural network for
    approximating Q values.

    Args:
        nb_actions: Size of action space, int
    """
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model


def get_agent(train=False):
    """
    Initializes environment, model, policy, and sequential memory
    for DQNAgent.

    Args:
        train: If False, loads BreakoutDeterministic-v0
        for testing else BreakoutDeterministic-v4

    Return: DQNAgent, ENV
    """
    if train:
        env = gym.make('BreakoutDeterministic-v4')
    else:
        env = 'BreakoutDeterministic-v0'
        env = gym.make(env, repeat_action_probability=0.0383)
    # print(env.env.)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    model = create_model(nb_actions)

    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)

    processor = Process()
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1.,
        value_min=0.1, value_test=.05, nb_steps=100000)

    dqn = MyDQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                     memory=memory, processor=processor,
                     nb_steps_warmup=5000, gamma=0.99,
                     target_model_update=5000, train_interval=4,
                     delta_clip=1.0)

    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    return dqn, env


if __name__ == "__main__":
    # Training Script
    # best model saved to weights_filename + "_best.h5"
    dqn, env = get_agent(train=True)
    weights_filename = './DQN_AtariBreakout/policy'
    log_filename = \
        './DQN_AtariBreakout/dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [FileLogger(log_filename, interval=10000)]
    dqn.fit(env, callbacks=callbacks,
            nb_steps=4000000, log_interval=25000,
            weights_filename=weights_filename)
    dqn.save_weights(weights_filename+".h5", overwrite=True)
    dqn.test(env, nb_episodes=1, visualize=False)
    env.close()
