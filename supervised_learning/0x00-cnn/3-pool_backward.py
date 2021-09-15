#!/usr/bin/env python3
"""
   Module contains
   pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
       Performs back propagation over a pooling layer of a neural network.

       Args:
         dA: numpy.ndarray - (m, hP, wP, cP) contains partial
           derivatives with respect to the pooling layer
         A_prev: numpy.ndarray - (m, hN, wN, c) output
           of previous layer
         kernel_shape: tuple - (kh, kw) size of kernel
         stride: tuple - (sh, sw) containing strides
         mode: 'max' or 'avg'

       Return:
         Partial derivatives with respect to previous layer.
    """
    m, hP, wP, cP = dA.shape
    _, hN, wN, cN = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    der = np.zeros_like(A_prev)

    for frame in range(m):
        for h in range(hP):
            ah = sh * h
            for w in range(wP):
                aw = sw * w
                for c in range(cP):
                    if mode == 'avg':
                        avg_dA = dA[frame, h, w, c] / kh / kw
                        der[frame, ah: ah+kh, aw: aw+kw, c] += (
                            np.ones((kh, kw)) * avg_dA
                        )
                    if mode == 'max':
                        box = A_prev[frame, ah: ah+kh, aw: aw+kw, c]
                        mask = (box == np.max(box))
                        der[frame, ah: ah+kh, aw: aw+kw, c] += (
                            mask * dA[frame, h, w, c]
                        )
    return der
