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
    w_Avgs, avg = [], 0
    for x, n in enumerate(data):
        avg = ((beta*avg) + ((1-beta)*n))
        bias_correct = avg / (1 - (beta**(x+1)))
        w_Avgs.append(bias_correct)
    return w_Avgs
