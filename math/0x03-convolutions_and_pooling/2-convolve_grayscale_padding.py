#!/usr/bin/env python3
"""
   Module contains
   convolve_grayscale_padding(images, kernel, padding):
"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
       Performs a valid convolution on grayscale images with custom padding.

       Args:
         images: numpy.ndarray - (m, h, w) contains multiple greyscale images
         kernel: numpy.ndarray - (kh, kw) kernel for convolution
         padding: (ph, pw) padding dimensions

       Returns:
         numpy.ndarray - convolved images
    """
    samples, samp_h, samp_w = images.shape
    filter_h, filter_w = kernel.shape
    pad_h, pad_w = padding

    samp_h = samp_h - filter_h + (2*pad_h) + 1
    samp_w = samp_w - filter_w + (2*pad_w) + 1

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
