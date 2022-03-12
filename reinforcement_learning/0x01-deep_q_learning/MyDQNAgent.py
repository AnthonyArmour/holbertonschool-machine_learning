#!/usr/bin/env python3
"""
This module contains a copy of keras-rl
DQGAgent for debuging and logging purposes.
"""


from rl.agents.dqn import DQNAgent
import warnings
from copy import deepcopy
import numpy as np
from tensorflow.keras.callbacks import History
from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)


class MyDQNAgent(DQNAgent):
    """
    This class wraps the DQNAgent class from
    https://github.com/taylormcnally/keras-rl2/blob/master/rl/core.py
    It is copied code that allows me to debug and log from the inside
    of the fit and test methods.
    """

    def fit(
            self,
            env,
            nb_steps,
            action_repetition=1,
            callbacks=None,
            verbose=1,
            visualize=False,
            nb_max_start_steps=0,
            start_step_policy=None,
            log_interval=10000,
            nb_max_episode_steps=None,
            render=False,
            weights_filename=None):
        """
        Trains the agent on the given environment.
        Read documentation for Agent class from
        keras-rl.
        """
        if action_repetition < 1:
            raise ValueError(
                f'action_repetition must be >= 1, is {action_repetition}')

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        track_reward = 0
        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the
                    # environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    assert observation is not None
                    if nb_max_start_steps == 0:
                        nb_random_start_steps = 0
                    else:
                        nb_random_start_steps = \
                            np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            ret = self.processor.process_step(
                                observation, reward, done, info)
                            observation, reward, done, info = ret
                        callbacks.on_action_end(action)
                        if done:
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = \
                                    self.processor.process_observation(
                                        observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    if render:
                        render.on_step_end(env)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        ret = self.processor.process_step(
                            observation, r, done, info)
                        observation, r, done, info = ret
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                ch = nb_max_episode_steps - 1
                if nb_max_episode_steps and episode_step >= ch:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    self.forward(observation)
                    self.backward(0., terminal=False)
                    if render:
                        render.on_episode_end(episode_reward)
                    if episode_reward > track_reward:
                        track_reward = episode_reward
                        self.save_weights(weights_filename +
                                          "_best.h5", overwrite=True)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            did_abort = True
            print("Safely Aborted")
            if render:
                return history, render
            return history
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        if render:
            return history, render
        return history

    def test(
            self,
            env,
            nb_episodes=1,
            action_repetition=1,
            callbacks=None,
            visualize=True,
            render=False,
            nb_max_episode_steps=None,
            nb_max_start_steps=5,
            start_step_policy=None,
            verbose=1,
            test_info=True):
        """
        Callback that is called before training begins.
        Read documentation for Agent class from
        keras-rl.
        """
        if action_repetition < 1:
            raise ValueError(
                f'action_repetition must be >= 1, is {action_repetition}')

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        # callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            if nb_max_start_steps == 0:
                nb_random_start_steps = 0
            else:
                nb_random_start_steps = \
                    np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, info, done = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        print("Custom Done")
                        break
                ch = nb_max_episode_steps - 1
                if nb_max_episode_steps and episode_step >= ch:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }

                render.on_step_end(env)
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history
