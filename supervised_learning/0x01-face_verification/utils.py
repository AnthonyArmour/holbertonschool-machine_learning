#!/usr/bin/env python3
"""
   Module contains
   load_images(images_path, as_array=True):
"""


import numpy as np
import cv2
import csv


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


def load_csv(csv_path, params={}):
    """
       Loads contents of csv as list of lists

       Args:
        csv_path: the path to csv to load
        params: parameters to load csv with

       Return:
        list of lists representing contents of csv
    """
    contents = []
    with open(csv_path, 'r') as fh:
        reader = csv.reader(fh, params)

        for row in reader:
            contents.append(row)

    return contents


def save_images(path, images, filenames):
    """
       Saves images to designated files

       Args:
        path: path to the directory in which the images should be saved
        images: list/numpy.ndarray of images to save
        filenames: list of filenames of the images to save

       Return:
        True on success, else False.
    """
    import os

    if os.path.exists(path):
        for x in range(len(images)):
            cv2.imwrite(path+"/"+filenames[x], images[x])
        return True
    else:
        return False
