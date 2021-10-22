#!/usr/bin/env python3
"""TrainModel class"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
import tensorflow.keras as K
from triplet_loss import TripletLoss


class TrainModel():
    """TrainModel class"""

    def __init__(self, model_path, alpha):
        """
           Class constructor

           Args:
            model_path: path to the base face verification embedding model
            alpha: alpha to use for the triplet loss calculation

           Creates new model.

        """
        target_shape = (96, 96)
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        anchor_input = Input(name="input_1", shape=target_shape + (3,))
        positive_input = Input(name="input_2", shape=target_shape + (3,))
        negative_input = Input(name="input_3", shape=target_shape + (3,))
        distances = TripletLoss(alpha)([
            self.base_model(anchor_input), self.base_model(positive_input),
            self.base_model(negative_input)
        ])
        self.training_model = K.Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=distances
        )
        self.training_model.compile(optimizer="adam")

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        history = self.training_model.fit(
            triplets, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, verbose=verbose
        )
        return history
