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
         dA: numpy.ndarray - (m, h_new, w_new, c_new) contains partial
           derivatives with respect to the pooling layer
         A_prev: numpy.ndarray - (m, h_prev, w_prev, c) output
           of previous layer
         kernel_shape: tuple - (kh, kw) size of kernel
         stride: tuple - (sh, sw) containing strides
         mode: 'max' or 'avg'

       Return:
         Partial derivatives with respect to previous layer.
    """
    _, hP, wP, cP = A_prev.shape
    m, hN, wN, cN = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    da = np.zeros_like(A_prev)

    for frame in range(m):
        for h in range(hN):
            ah = sh*h
            for w in range(wN):
                aw = sw+w
                for flt in range(cN):
                    if mode == 'avg':
                        avg = dA[frame, h, w, flt]/kh/kw
                        da[frame, ah:ah+kh, aw:aw+kw, flt] += (
                            np.ones((kh, kw))*avg
                        )
                    else:
                        box = A_prev[frame, ah:ah+kh, aw:aw+kw, flt]
                        mask = (box == np.max(box))
                        da[frame, ah:ah+kh, aw:aw+kw, flt] += (
                            mask*dA[frame, h, w, flt]
                        )

    return da
