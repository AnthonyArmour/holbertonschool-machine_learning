#!/usr/bin/env python3
"""
Module contains function for creating variational autoencoder.
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates variational autoencoder network.

    Args:
        input_dims: Integer containing the dimensions of the model input
        hidden_layers: List containing the number of nodes for each hidden
        layer in the encoder, respectively
        latent_dims: Integer containing the dimensions of the latent space
        representation

        Return: encoder, decoder, auto
        encoder: Encoder model, which should output the latent representation,
        the mean, and the log variance, respectively
        decoder: Decoder model
        auto: Full autoencoder model
    """
    pass
