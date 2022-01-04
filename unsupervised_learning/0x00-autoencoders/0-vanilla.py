#!/usr/bin/env python3
"""
Module contains function that creates an autoencoder network.
"""


import tensorflow.keras as keras
from tensorflow.keras import activations


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder network.

    Args:
        input_dims: Integer - Dimensions of the model input.
        hidden_layers: List - Number of nodes for each hidden layer in
        the encoder.
        latent_dims: Integer - Dimensions of the latent space representation.

    Return: encoder, decoder, autoencoder
        encoder: Encoder model.
        decoder: Decoder model.
        autoencoder: Full autoencoder model.
    """

    Dense = keras.layers.Dense

    input_layer = keras.Input(shape=(input_dims,))
    encoded_input = keras.Input(shape=(latent_dims,))

    encoded_layers = Dense(hidden_layers[0], activation="relu")(input_layer)

    for nodes in hidden_layers[1:]+[latent_dims]:
        encoded_layers = Dense(nodes, activation="relu")(encoded_layers)

    decoded = Dense(hidden_layers[-1], activation="relu")(encoded_layers)

    for x, nodes in enumerate(list(reversed(hidden_layers[:-1]))+[input_dims]):
        if x == len(hidden_layers) - 1:
            decoded = Dense(nodes, activation="sigmoid")(decoded)
        else:
            decoded = Dense(nodes, activation="relu")(decoded)

    autoencoder = keras.Model(input_layer, decoded)

    encoder = keras.Model(input_layer, encoded_layers)

    decoder_layer = encoded_input
    for layer in autoencoder.layers[len(hidden_layers)+2:]:
        decoder_layer = layer(decoder_layer)

    decoder = keras.Model(encoded_input, decoder_layer)

    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
