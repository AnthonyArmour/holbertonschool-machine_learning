#!/usr/bin/env python3
"""TrainModel class"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
import tensorflow.keras as K
import tensorflow.keras.backend as backend
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
        """Training Method"""
        history = self.training_model.fit(
            triplets, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, verbose=verbose
        )
        return history

    def save(self, save_path):
        """Saves mode to save_path"""
        self.training_model.save(save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """
           Calculates F1 score:
            2*((precision*recall)/(precision+recall))

           Args:
            y_true: numpy.ndarray of shape (m,) -
              containing the correct labels
                m: number of examples
            y_pred: numpy.ndarray of shape (m,) -
              containing the predicted labels

           Return:
            F1_score
        """

        y_true = tf.convert_to_tensor(value=y_true, dtype='float32')
        y_pred = tf.convert_to_tensor(value=y_pred, dtype='float32')

        true_positives = backend.sum(backend.clip(y_true * y_pred, 0, 1))
        possible_positives = backend.sum(backend.clip(y_true, 0, 1))
        recall = true_positives / (possible_positives + backend.epsilon())

        true_positives = backend.sum(backend.clip(y_true * y_pred, 0, 1))
        predicted_positives = backend.sum(backend.clip(y_pred, 0, 1))
        precision = true_positives / (predicted_positives + backend.epsilon())

        F1_score = 2*((precision*recall)/(precision+recall))

        return tf.Session().run(F1_score)

    @staticmethod
    def accuracy(y_true, y_pred):
        """
           Calculates Accuracy

           Args:
            y_true: numpy.ndarray of shape (m,) -
              containing the correct labels
                m: number of examples
            y_pred: numpy.ndarray of shape (m,) -
              containing the predicted labels

           Return:
            Accuracy
        """

        accuracy = sum(map(
            lambda x, y: x == y == 1, y_true, y_pred
            ))/sum(y_true)
        return accuracy

    def best_tau(self, images, identities, thresholds):
        """
           Calculates the best tau to use for a maximal F1 score.
        """

        tf.keras.me