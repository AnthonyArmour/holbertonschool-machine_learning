#!/usr/bin/env python3
"""
   Module contains
   conv_forward(
       A_prev, W, b, activation, padding="same", stride=(1, 1)
       ):
"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
       Performs forward propagation over a convolutional
         layer of a neural network.

       Args:
         A_prev: numpy.ndarray - (m, h_prev, w_prev, c_prev) output
           of previous layer
         W: numpy.ndarray - kernels for convolution
         b: numpy.ndarray - bias of shape(1, 1, 1, c_new)
         activation: activation function
         padding: Indicates type of padding
         stride: tuple - (sh, sw) containing strides
    """
    m, hP, wP, cP = A_prev.shape
    kh, kw, cP, cN = W.shape
    sh, sw = stride

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((hP - 1) * sh) + kh - hP) // 2
        pad_w = (((wP - 1) * sw) + kw - wP) // 2

    padded = np.pad(
        A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant', constant_values=0
        )

    outH = (hP + (2 * pad_h) - kh) // sh + 1
    outW = (wP + (2 * pad_w) - kw) // sw + 1

    conv = np.zeros((m, outH, outW, cN))

    for prop in range(cN):
        for i in range(outH):
            for j in range(outW):
                h = i * sh
                w = j * sw
                conv[:, i, j, prop] = np.sum(np.multiply(
                    W[:, :, :, prop], padded[:, h:h+kh, w:w+kw]
                    ), axis=(1, 2, 3)) + b[:, :, :, prop]

    return activation(conv)
