#!/usr/bin/env python3
"""Module contains moving_average(data, beta) function"""


import numpy as np


def moving_average(data, beta):
    """
       Calculates the weighted moving average of a data set.

       Args:
         data: list of data to calculate the moving average of
         beta: weight used for the moving average.

       Returns:
         A list containing the moving averages of data.
    """
