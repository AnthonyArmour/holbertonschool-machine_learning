#!/usr/bin/env python3
"""FaceVerification class"""


import tensorflow as tf
import tensorflow.keras as K


class FaceVerification():
    """Face Verification Class"""

    def __init__(self, model_path, database, identities):
        """
           Class Constructor

           Public Attributes:
                model_path: path to where the face verification
                embedding model is stored
                database: numpy.ndarray of shape (d, e) -
                containing all the face embeddings in the database
                    d: number of images in the database
                    e: dimensionality of the embedding
                identities: list of length d containing the
                identities corresponding to the embeddings in database
        """

        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model_path = K.models.load_model(model_path)
        self.database = database
        self.identities = identities
