#!/usr/bin/env python3
"""
   Module contains
   load_images(images_path, as_array=True):
"""


import numpy as np
import imageio


def load_images(images_path, as_array=True):
    """

       Args:
         images_path: the path to a directory from which to load images
         as_array: boolean indicating whether the images
           should be loaded as one numpy.ndarray

       Return:
         images, filenames

    """
    import os
    images_paths = os.listdir(images_path)
    images, filenames = [], []

    for path in sorted(images_paths):
        images.append(imageio.imread(images_path+"/"+path))
        filenames.append(path)

    if as_array:
        images = np.stack(images, axis=0)

    return images, filenames
