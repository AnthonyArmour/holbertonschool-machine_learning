#!/usr/bin/env python3
"""
   Module contains:
   Class Yolo
"""


import tensorflow.keras as K
import tensorflow.keras.backend as backend
import tensorflow as tf
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

    def process_outputs(self, outputs, image_size):
        """
           Args:
             outputs: numpy.ndarray - contains predictions from model
               for single image.
             image_size: numpy.ndarray - images original
               size (img_height, img_width)

           Return:
              tuple - (boxes, box_confidence, box_class_probs)
              boxes: numpy.ndarray - (grid_height, grid_width, anchorboxes, 4)
                 4 => (x1, y1, x2, y2)
              box_confidence: numpy.ndarray - shape
                (grid_height, grid_width, anchor_boxes, 1)
              box_class_probs: numpy.ndarray - shape
                (grid_height, grid_width, anchor_boxes, classes)
                contains class probabilities for each output
        """
        img_height, img_width = image_size[0], image_size[1]
        boxes = [output[:, :, :, :4] for output in outputs]
        # print("Boxes shape - {} {} {}".format(
        # boxes[0].shape, boxes[1].shape, boxes[2].shape))
        box_confidence = []
        class_probs = []
        # box_confidence = [output[:, :, :, 5] for output in outputs]
        # class_probs = [output[:, :, :, 6:] for output in outputs]
        Cx, Cy, cornersX, cornersY = [], [], [], []
        for output in outputs:
            Cx.append(img_width/output.shape[1])
            Cy.append(img_height/output.shape[0])
            cornersX.append(np.zeros((output.shape[0], output.shape[1], 1)))
            cornersY.append(np.zeros((output.shape[0], output.shape[1], 1)))

        for i in range(len(cornersX)):
            for x in range(outputs[i].shape[0]):
                for y in range(outputs[i].shape[1]):
                    cornersX[i][x, y, 0] = Cx[i] * y
                    cornersY[i][x, y, 0] = Cy[i] * x

        # print("CornersX then CornersY")
        # for x in range(3):
        #     print("shape", cornersX[x].shape, cornersY[x].shape)
        # print(cornersX)
        # print(cornersY)

        for output in outputs:
            box_confidence.append(backend.sigmoid(output[:, :, :, 5]))

        for output in outputs:
            class_probs.append(backend.softmax(output[:, :, :, 6:]))

        sess = backend.get_session()
        for x, box in enumerate(boxes):
            b1 = (
                (backend.sigmoid(box[:, :, :, 0])+cornersX[x])/img_width
                )
            # print("b1 and folling shapes")
            # print(b1.shape)
            b2 = (
                (backend.sigmoid(box[:, :, :, 1])+cornersY[x])/img_height
                )
            # print(b2.shape)
            b3 = (
                (tf.math.exp(box[:, :, :, 2])*self.anchors[x, :, 0])/img_width
                )
            # print(b3.shape)
            b4 = (
                (tf.math.exp(box[:, :, :, 3])*self.anchors[x, :, 1])/img_height
                )
            # print(type(sess.run(b4)))
            box = np.stack(
                (sess.run(b1), sess.run(b2), sess.run(b3), sess.run(b4)),
                axis=2)
            # print(box.shape, "box shape")
        box_pred = []

        for box in boxes:
            x1 = box[:, :, :, 0] - (box[:, :, :, 2] * 0.5)
            y1 = box[:, :, :, 1] - (box[:, :, :, 3] * 0.5)
            x2 = box[:, :, :, 0] + (box[:, :, :, 2] * 0.5)
            y2 = box[:, :, :, 1] + (box[:, :, :, 3] * 0.5)
            box_pred.append(np.concatenate((x1, y1, x2, y2), axis=2))

        # for box in box_pred:
        #     print("prediction box shape- ", box.shape)

        return (box_pred, sess.run(box_confidence), sess.run(class_probs))
