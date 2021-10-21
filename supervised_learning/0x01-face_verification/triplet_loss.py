#!/usr/bin/env python3
"""contains TripletLoss class"""


import numpy as np
import tensorflow as tf
import tensorflow.keras as K


class TripletLoss(K.layers.Layer):
    """TripletLoss class"""

    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        """Class constructor"""
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """
           Computes triplet loss

           Args:
            inputs: list containing the anchor, positive
              and negative output tensors from the last
              layer of the model, respectively.

           Return:
            tensor containing losses.
        """
        dist_ap = np.linalg.norm(inputs[0]-inputs[1], axis=1)
        dist_an = np.linalg.norm(inputs[0]-inputs[2], axis=1)
        return tf.maximum(dist_ap - dist_an + self.alpha, 0.0)
