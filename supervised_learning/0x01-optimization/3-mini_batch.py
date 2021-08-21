#!/usr/bin/env python3
"""Module contains train_mini_batch() function"""


import numpy as np


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
       Trains a loaded neural network model using
       mini-batch gradient descent

       Args:
         X_train: numpy.ndarray - (m, 784) containing training data
         Y_train: numpy.ndarray - (m, 10) one-hot containing traing labels
         X_valid: numpy.ndarray - (m, 784) containing validation data
         Y_valid: numpy.ndarray - (m, 10) one-hot containing validation labels
         batch_size: number of data points in a batch
         epochs: training iteration over entire dataset
         load_path: path from which to load the model
         save_path: path to where the model should be saved after training

       Returns:
         The path where the model was saved.
    """
