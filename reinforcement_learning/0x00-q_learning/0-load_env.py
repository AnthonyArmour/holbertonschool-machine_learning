#!/usr/bin/env python3
"""
Module contains function for creating
open ai gym FrozenLake environment.
"""


import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLakeEnv from open ai gym.

    Args:
        desc: None or list of lists containing a custom
        description of the map to load for the environment.
        map_name: None or string containing the
        pre-made map to load.
        is_slippery: boolean to determine if the ice is slippery

    Return: The environment
    """
    env = gym.envs.make(
        "FrozenLake-v1", desc=desc, map_name=map_name,
        is_slippery=is_slippery)
    return env
