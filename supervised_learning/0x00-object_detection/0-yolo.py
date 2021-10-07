#!/usr/bin/env python3
"""
   Module contains:
   Class Yolo
"""


import tensorflow.keras as K
import numpy as np


class Yolo():
    """
       Yolo v3 class for performing object detection.

       Public Instance Attributes
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
           Init method for instanciating Yolo class.

           Args:
             model_path: path to Darknet keras model
             classes_path: path to list of class names for
               darknet model
             class_t: float representing box score for initial
               filtering step
             nms_t: float representing IOU threshold for non-max
               supression
             anchors: numpy.ndarray - shape (outputs, anchor_boxes, 2)
               containing all anchor boxes
                 outputs: number of predictions made
                 anchor_boxes: number of anchor boxes for each pred.
                 2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'rt') as fd:
            self.class_names = fd.read().rstrip('\n').split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
