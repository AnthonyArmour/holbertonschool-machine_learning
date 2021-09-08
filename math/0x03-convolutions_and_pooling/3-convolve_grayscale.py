#!/usr/bin/env python3
"""
   Module contains
   convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
       Performs a valid convolution on grayscale images.

       Args:
         images: numpy.ndarray - (m, h, w) contains multiple greyscale images
         kernel: numpy.ndarray - (kh, kw) kernel for convolution
         padding: (ph, pw) tuple of padding dimensions
         stride: (sh, sw) tuple of stride dimensions

       Returns:
         numpy.ndarray - convolved images
    """
    samples, samp_h, samp_w = images.shape
    filter_h, filter_w = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        pad_h, pad_w = padding
        samp_h = samp_h - filter_h + (2*pad_h) + 1
        samp_w = samp_w - filter_w + (2*pad_w) + 1
    elif padding == "valid":
        pad_h, pad_w = 0, 0
    elif padding == "same":
        pad_h = (((samp_h - 1) * sh) + filter_h - samp_h) // 2 + 1
        pad_w = (((samp_w - 1) * sw) + filter_w - samp_w) // 2 + 1

    images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant'
        )

    samp_h = (samp_h + (2 * pad_h) - filter_h) // sh + 1
    samp_w = (samp_w + (2 * pad_w) - filter_w) // sw + 1

    conv = np.zeros((samples, samp_h, samp_w))

    for w in range(samp_h):
        for h in range(samp_w):
            conv[:, w, h] = (
                kernel * images[:, sw*w: sw*w+filter_h, sh*h:sh*h+filter_w]
                ).sum(axis=(1, 2))
    return conv
