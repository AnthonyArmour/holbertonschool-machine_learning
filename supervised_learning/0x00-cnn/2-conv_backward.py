#!/usr/bin/env python3
"""
   Module contains
   conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
       Performs back propagation over a convolutional layer of
         a neural network.

       Args:
         dZ: numpy.ndarray - (m, h_new, w_new, c_new) contains partial
           derivatives with respect to non activated conv layer
         A_prev: numpy.ndarray - (m, h_prev, w_prev, c_prev) output
           of previous layer
         W: numpy.ndarray - kernels for convolution
         b: numpy.ndarray - bias of shape(1, 1, 1, c_new)
         padding: Indicates type of padding
         stride: tuple - (sh, sw) containing strides
    """
    m, hP, wP, cP = A_prev.shape
    m, hN, wN, cN = dZ.shape
    kh, kw, cP, cN = W.shape
    sh, sw = stride

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((hP - 1) * sh) + kh - hP) // 2
        pad_w = (((wP - 1) * sw) + kw - wP) // 2

    A_prev = np.pad(
        A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant', constant_values=0
        )

    dW = np.zeros_like(W)
    dA = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for frame in range(m):
        for h in range(hN):
            for w in range(wN):
                for flt in range(cN):
                    filter = W[:, :, :, flt]
                    dz = dZ[frame, h, w, flt]
                    box = A_prev[frame, h*sh:h*sh+kh, w*sw:w*sw+kw, :]
                    dW[:, :, :, flt] += box*dz
                    dA[frame, h*sh:h*sh+kh, w*sw:w*sw+kw, :] += dz*filter

    if padding == 'same':
        dA = dA[:, pad_h:-pad_h, pad_w:-pad_w, :]
    return dA, dW, db
