#!/usr/bin/env python3
"""contains TripletLoss class"""


import numpy as np
import tensorflow.keras as K


class TripletLoss(K.layers.Layer):
    """TripletLoss class"""

    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        """Class constructor"""
        self.alpha = alpha
