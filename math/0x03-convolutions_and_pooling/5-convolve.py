#!/usr/bin/env python3
"""
   Module contains
   convolve(images, kernels, padding='same', stride=(1, 1)):
"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
       Performs a convolution on images using multiple kernels.

       Args:
         images: numpy.ndarray - (m, h, w) contains multiple greyscale images
         kernel: numpy.ndarray - (kh, kw, c, nc) kernels for convolution
         padding: (ph, pw) tuple of padding dimensions
         stride: (sh, sw) tuple of stride dimensions

       Returns:
         numpy.ndarray - convolved images
    """
    samples, samp_h, samp_w, samp_c = images.shape
    filter_h, filter_w, filter_d, nc = kernels.shape
    sh, sw = stride

    if type(padding) is tuple:
        pad_h, pad_w = padding
    elif padding == "valid":
        pad_h, pad_w = 0, 0
    elif padding == "same":
        pad_h = (((samp_h - 1) * sh) + filter_h - samp_h) // 2 + 1
        pad_w = (((samp_w - 1) * sw) + filter_w - samp_w) // 2 + 1

    padded = np.pad(
        images, ((0,), (pad_h,), (pad_w,), (0,)), 'constant'
        )

    out_h = (samp_h + (2 * pad_h) - filter_h) // sh + 1
    out_w = (samp_w + (2 * pad_w) - filter_w) // sw + 1

    conv = np.zeros((samples, out_h, out_w, nc))

    for filterx in range(nc):
        for h in range(out_h):
            for w in range(out_w):
                conv[:, h, w, filterx] = np.sum(
                    np.multiply(
                        kernels[:, :, :, filterx],
                        padded[:, sh*h: sh*h+filter_h, sw*w:sw*w+filter_w]
                    ), axis=(1, 2, 3)
                )
    return conv
