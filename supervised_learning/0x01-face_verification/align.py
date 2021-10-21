#!/usr/bin/env python3
"""
   Module contains class FaceAlign
"""


import numpy as np
import dlib


class FaceAlign():
    """Face Alignment Class"""

    def __init__(self, shape_predictor_path):
        """
           Class Constructor

           Args:
            shape_predictor_path: path to dlib shape predictor model.

           Attributes:
            detector: contains dlib's default face detector
            shape_predictor: contains dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
           Detects a face in an image.

           Args:
            image: rank 3 np.ndarray - containing image

           Return:
            dlib.rectangle - containing boundary box or None.
                If multiple faces are detected, return the dlib.rectangle
                  with the largest area.
                If no faces are detected, return a dlib.rectangle that
                  is the same as the image.
        """
        def area(x):
            """gets area of box"""
            return (x.right()-x.left())*(x.bottom()-x.top())

        dets = self.detector(image, 1)
        if len(dets):
            # Get detection with largest box
            dets = sorted(dets, key=lambda x: area(x), reverse=True)
            return dets[0]
        else:
            return dlib.rectangle(0, 0, image.shape[-2], image.shape[-3])