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
