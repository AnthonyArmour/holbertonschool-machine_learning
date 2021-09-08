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
    elif padding == "valid":
        pad_h, pad_w = 0, 0
    elif padding == "same":
        pad_h = (((samp_h - 1) * sh) + filter_h - samp_h) // 2 + 1
        pad_w = (((samp_w - 1) * sw) + filter_w - samp_w) // 2 + 1

    padded = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant'
        )

    out_h = (samp_h + (2 * pad_h) - filter_h) // sh + 1
    out_w = (samp_w + (2 * pad_w) - filter_w) // sw + 1

    conv = np.zeros((samples, samp_h, samp_w))

    for h in range(out_h):
        for w in range(out_w):
            conv[:, w, h] = (
                kernel * padded[:, sh*h: sh*h+filter_h, sw*w:sw*w+filter_w]
                ).sum(axis=(1, 2))
    return conv
