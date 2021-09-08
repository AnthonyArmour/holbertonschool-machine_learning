#!/usr/bin/env python3
"""
   Module contains
   convolve_grayscale_same(images, kernel):
"""


import numpy as np
from math import ceil, floor


def convolve_grayscale_same(images, kernel):
    """
       Performs a valid convolution on grayscale images.

       Args:
         images: numpy.ndarray - (m, h, w) contains multiple greyscale images
         kernel: numpy.ndarray - (kh, kw) kernel for convolution

       Returns:
         numpy.ndarray - convolved images
    """
    samples, samp_h, samp_w = images.shape
    filter_h, filter_w = kernel.shape

    pad_h = ceil((filter_h-1)/2)
    pad_w = ceil((filter_w-1)/2)

    padded = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant'
        )

    conv = np.zeros((samples, samp_h, samp_w))

    for w in range(samp_h):
        for h in range(samp_w):
            conv[:, w, h] = (
                kernel * padded[:, w: w+filter_h, h:h+filter_w]
                ).sum(axis=(1, 2))
    return conv
