#!/usr/bin/env python3
"""
   Module contains
   load_images(images_path, as_array=True):
"""


import numpy as np
import cv2


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
        image = cv2.imread(images_path+"/"+path)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        filenames.append(path)

    if as_array:
        images = np.stack(images)

    return images, filenames
