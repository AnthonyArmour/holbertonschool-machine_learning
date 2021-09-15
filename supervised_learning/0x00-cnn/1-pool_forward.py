#!/usr/bin/env python3
"""
   Module contains
   pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
       Performs forward propagation over a pooling layer of a
         neural network.

       Args:
         A_prev: numpy.ndarray - (m, h_prev, w_prev, c_prev) output
           of previous layer
         kernel_shape: tuple - (kh, kw) contains size of kernel for
           pooling
         stride: tuple - (sh, sw) containing strides
         mode: indicates whether to use max or avg
    """
    m, hP, wP, cP = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    outH = int(((hP - kh) / sh) + 1)
    outW = int(((wP - kw) / sw) + 1)

    conv = np.zeros((m, outH, outW, cP))

    for i in range(outH):
        for j in range(outW):
            h = i * sh
            w = j * sw
            box = A_prev[:, h:h+kh, w:w+kw, :]
            if mode == 'max':
                conv[:, i, j, :] = np.max(box, axis=(1, 2))
            else:
                conv[:, i, j, :] = np.mean(box, axis=(1, 2))

    return conv
