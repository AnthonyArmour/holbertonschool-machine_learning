#!/usr/bin/env python3
"""
   Module contains
   load_images(images_path, as_array=True):
"""


import numpy as np
import cv2
import csv

debug = [
    'CarlosArias1.jpg', 'CarlosArias2.jpg',
    'CarlosArias3.jpg', 'CarlosArias4.jpg',
    'ChristianWilliams2.jpg', 'DavidLatorre.jpg',
    'DennisPham6.jpg', 'ElaineYeung2.jpg',
    'ElaineYeung3.jpg', 'FeliciaHsieh.jpg',
    'HongtuHuang.jpg', 'JavierCañon6.jpg',
    'JohnCook.jpg', 'JosefGoodyear.jpg',
    'JuanValencia4.jpg', 'KennethCortesAguas1.jpg',
    'MohamethSeck1.jpg', 'MohamethSeck3.jpg',
    'PhuTruong3.jpg', 'PhuTruong4.jpg',
    'TimAssavarat4.jpg', 'YesidGonzalez0.jpg',
    'YesidGonzalez2.jpg'
    ]


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
    import matplotlib.image as mpimg
    images_paths = os.listdir(images_path)
    images, filenames = [], []
    d = []

    for path in sorted(images_paths):
        image = cv2.imread(images_path+"/"+path)
        if image is None:
            continue

        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        filenames.append(path)

    if as_array:
        images = np.array(images)

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


def generate_triplets(images, filenames, triplet_names):
    """
       Generates triplets

       Args:
        images: numpy.ndarray of shape (i, n, n, 3) containing
          the aligned images in the dataset
            i: number of images
            n: size of the aligned images
        filenames: list of length i containing the corresponding
          filenames for images
        triplet_names: list of length m of lists where each sublist
          contains the filenames of an anchor, positive,
          and negative image, respectively

       Return:
        list - [A, P, N]
            A is a numpy.ndarray of shape (m, n, n, 3) -
              containing the anchor images for all m triplets
            P is a numpy.ndarray of shape (m, n, n, 3) -
              containing the positive images for all m triplets
            N is a numpy.ndarray of shape (m, n, n, 3) -
              containing the negative images for all m triplets
    """
    imgs = {}
    A, P, N = [], [], []

    for x, file in enumerate(filenames):
        imgs[file] = images[x]

    for triplet in triplet_names:
        if (triplet[0]+".jpg" in filenames and
           triplet[1]+".jpg" in filenames and
           triplet[2]+".jpg" in filenames):
            A.append(imgs[triplet[0]+".jpg"])
            P.append(imgs[triplet[1]+".jpg"])
            N.append(imgs[triplet[2]+".jpg"])

    return [np.stack(A), np.stack(P), np.stack(N)]
