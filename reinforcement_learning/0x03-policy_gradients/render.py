#!/usr/bin/env python3
"""
Module contains class for rendering
game play of CartPole environment.
"""


import matplotlib.pyplot as plt


class Render():
    """
    Class for rendering frames during play.
    """

    def __init__(self):
        """Class constructor"""
        self.ims = []

    def on_step(self, env):
        """Saves frame at each step"""
        im = plt.imshow(env.render(mode='rgb_array'))
        self.ims.append([im])
