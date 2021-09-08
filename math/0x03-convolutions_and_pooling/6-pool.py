#!/usr/bin/env python3
"""
   Module contains
   pool(images, kernel_shape, stride, mode='max'):
"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
       Performs pooling on images.

       Args:
         images: numpy.ndarray - (m, h, w, c) contains multiple
           greyscale images
         kernel_shape: numpy.ndarray - (kh, kw) kernels for convolution
         stride: (sh, sw) tuple of stride dimensions
         mode: type of pooling

       Returns:
         numpy.ndarray - convolved images
    """
    samples, samp_h, samp_w, samp_c = images.shape
    fh, fw = kernel_shape
    sh, sw = stride

    if mode == "max":
        op = np.amax
    else:
        op = np.average

    pad_h, pad_w = 0, 0

    out_h = (samp_h + (2 * pad_h) - fh) // sh + 1
    out_w = (samp_w + (2 * pad_w) - fw) // sw + 1

    conv = np.zeros((samples, out_h, out_w, samp_c))

    for h in range(out_h):
        for w in range(out_w):
            conv[:, h, w, :] = op(
                images[:, sh*h: sh*h+fh, sw*w:sw*w+fw, :], axis=(1, 2)
                )
    return conv
