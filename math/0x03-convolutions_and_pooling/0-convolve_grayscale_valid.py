#!/usr/bin/env python3
"""
   Module contains
   convolve_grayscale_valid(images, kernel):
"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
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

    dim_h = samp_h - filter_h + 1
    dim_w = samp_w - filter_w + 1

    conv = np.zeros((samples, dim_h, dim_w))

    for w in range(dim_w):
        for h in range(dim_h):
            conv[:, h, w] = (
                kernel * images[:, h:h+filter_h, w: w+filter_w]
                ).sum()
    return conv
