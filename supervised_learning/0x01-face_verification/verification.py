#!/usr/bin/env python3
"""FaceVerification class"""


import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
import tensorflow.keras as K
import tensorflow.keras.backend as backend
from triplet_loss import TripletLoss
# import utils
# from train_model import TrainModel


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

        with CustomObjectScope({'tf': tf, "TripletLoss": TripletLoss}):
            self.model = K.models.load_model(model_path)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
           Generates embeddings of images.

           Args:
            images: numpy.ndarray of shape (i, n, n, 3) -
              containing the aligned images
                    i: number of images
                    n: size of the aligned images

           Return:
                embedding of shape (images, logits)
        """
        embedding = self.model.predict(images)
        return embedding

    def verify(self, image, tau=0.5):
        """
           Makes verification prediction

           Args:
                image: numpy.ndarray of shape (n, n, 3) -
                  containing the aligned image of the face to be verified
                tau: maximum euclidean distance used for verification

           Returns:
                (identity, distance), or (None, None) on failure

                identity: string containing the identity
                  of the verified face
                distance: euclidean distance between the verified
                  face embedding and the identified database embedding
        """
        logits = self.embedding(image[np.newaxis, ...])
        distances = []
        for x in range(self.database.shape[0]):
            distances.append(
                tf.Session().run(
                    tf.reduce_sum(tf.square((self.database[x]-logits)), axis=1)
                )[0]
            )
        idx = np.argmin(distances)
        if distances[idx] <= tau:
            return (self.identities[idx], distances[idx])
        return (None, None)
