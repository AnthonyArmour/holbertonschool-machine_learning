#!/usr/bin/env python3
"""
   Module contains
   def model(Data_train, Data_valid, layers,
   activations, alpha=0.001, beta1=0.9, beta2=0.999,
   epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
   save_path='/tmp/model.ckpt'):
"""


import numpy as np


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
       Builds, trains, and saves a neural network model in tensorflow
       using Adam optimization, mini-batch gradient descent, learning
       rate decay, and batch normalization.

       Args:
         Data_train: tuple containing the training inputs and
           training labels, respectively
         Data_valid: tuple containing the validation inputs and
           validation labels, respectively
         layers: list containing the number of nodes in each layer of
           the network
         activation: list containing the activation functions used for
           each layer of the network
         alpha: learning rate
         beta1: weight for the first moment of Adam Optimization
         beta2: weight for the second moment of Adam Optimization
         epsilon: small number used to avoid division by zero
         decay_rate: decay rate for inverse time decay of the learning rate
         batch_size: number of data points that should be in a mini-batch
         epochs: number of times the training should pass through
           the whole dataset
         save_path: path where the model should be saved to

       Returns:
         The path where the model was saved.
    """
