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

    def sigmoid(self, arr):
        """sigmoid activation function"""
        return 1 / (1+np.exp(-1*arr))

    def process_outputs(self, outputs, image_size):
        """
           Args:
             outputs: numpy.ndarray - contains predictions from model
               for single image.
             image_size: numpy.ndarray - images original
               size (image_height, image_width)

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
        IH, IW = image_size[0], image_size[1]
        boxes = [output[:, :, :, :4] for output in outputs]
        box_confidence = []
        class_probs = []

        cornersX, cornersY = [], []
        for output in outputs:
            # Get width and height of grid cells
            gridH, gridW, anchors = output.shape[:3]
            cx = np.arange(gridW).reshape(1, gridW)
            cx = np.repeat(cx, gridH, axis=0)
            cy = np.arange(gridW).reshape(1, gridW)
            cy = np.repeat(cy, gridH, axis=0).T
            # print("\n\n\n\n\n\n\ncx shape- {} | cy shape- {}
            # \n\n\n\n\n".format(cx.shape, cy.shape))
            cornersX.append(
                np.repeat(cx[..., np.newaxis], anchors, axis=2)
                )
            cornersY.append(
                np.repeat(cy[..., np.newaxis], anchors, axis=2)
                )
            box_confidence.append(self.sigmoid(output[..., 4:5]))
            class_probs.append(self.sigmoid(output[..., 5:]))

            # create grid cells
            # cornersX.append(np.zeros((output.shape[0], output.shape[1], 1)))
            # cornersY.append(np.zeros((output.shape[0], output.shape[1], 1)))

            # cx = np.arange(grid_width).reshape(1, grid_width)
            # cx = np.repeat(cx, grid_height, axis=0)
            # cx = np.repeat(cx[..., np.newaxis], anchors, axis=2)
            # cy = np.arange(grid_width).reshape(1, grid_width)
            # cy = np.repeat(cy, grid_height, axis=0).T
            # cy = np.repeat(cy[..., np.newaxis], anchors, axis=2)

        # Set grid cells top left corner X and Y

        # for i in range(len(cornersX)):
        #     for k in range(outputs[i].shape[0]):
        #         for j in range(outputs[i].shape[1]):
        #             cornersX[i][k, j, 0] = Cx[i] * j
        #             cornersY[i][k, j, 0] = Cy[i] * k

        # sess = backend.get_session()

        # for out in outputs:
        #     # conf = sess.run(backend.sigmoid(out[:, :, :, 0]))
        #     conf = self.sigmoid(out[:, :, :, 0])
        #     # box_confidence.append(np.expand_dims(conf, axis=2))
        #     shp = out.shape[:3]
        #     box_confidence.append(self.sigmoid(out[..., 4:5]))
        #     class_probs.append(self.sigmoid(out[..., 5:]))

        for x, box in enumerate(boxes):
            b1 = (
                (self.sigmoid(box[:, :, :, 0])+cornersX[x])/IW
                )
            b2 = (
                (self.sigmoid(box[:, :, :, 1])+cornersY[x])/IH
                )
            b3 = (
                (np.exp(box[:, :, :, 2])*self.anchors[x, :, 0])/IW
                )
            b4 = (
                (np.exp(box[:, :, :, 3])*self.anchors[x, :, 1])/IH
                )
            box = np.stack((b1, b2, b3, b4), axis=2)
            # print(box.shape, "box shape")
            # (sess.run(b1), sess.run(b2), sess.run(b3), sess.run(b4))
        box_pred = []

        for box in boxes:
            x1 = box[..., 0] - (box[..., 2] * 0.5)*IW
            y1 = box[..., 1] - (box[..., 3] * 0.5)*IH
            x2 = box[..., 0] + (box[..., 2] * 0.5)*IW
            y2 = box[..., 1] + (box[..., 3] * 0.5)*IH
            box_pred.append(np.concatenate((x1, y1, x2, y2), axis=2))

        return (box_pred, box_confidence, class_probs)
