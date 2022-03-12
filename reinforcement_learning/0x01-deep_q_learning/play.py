#!/usr/bin/env python3
"""
Module contains methods for testing and
rendering a match played by a trained
DQNAgent for atari breakout.
"""


from train import (
    get_agent,
    ENV_NAME,
    INPUT_SHAPE,
    WINDOW_LENGTH)
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import matplotlib.animation as animation
from IPython.display import HTML


class Render():
    """
    Class for rendering frames during play.
    """

    def __init__(self):
        """Class constructor"""
        self.ims = []
        self.best = []
        self.reward = 0

    def on_step_end(self, env):
        """Saves frame at each step"""
        im = plt.imshow(env.render(mode='rgb_array'))
        self.ims.append([im])


def test(dqn, env):
    """
    Function for testing and saving played match
    by trained DQNAgent.

    Args:
        dqn: Trained DQNAgent
        env: Atari gym environment
    """
    callbacks = Render()
    fig = plt.figure(figsize=(4, 5))
    plt.axis('off')
    dqn.test(env, nb_episodes=1, visualize=False,
             render=callbacks, nb_max_episode_steps=4000)
    print(len(callbacks.ims))
    return fig, callbacks.ims


if __name__ == "__main__":
    # Testing Script
    dqn, env = get_agent()
    weights_filename = './DQN_AtariBreakout/policy_best.h5'
    dqn.load_weights(weights_filename)
    fig, ims = test(dqn, env)
    ani = animation.ArtistAnimation(fig=fig, artists=ims, interval=20)
    print("Save? ", end="")
    save = input()
    if "y" in save:
        ani.save("./DQN_AtariBreakout/anim.gif", writer="pillow")
    plt.close()
    env.close()
    HTML(ani.to_jshtml())
