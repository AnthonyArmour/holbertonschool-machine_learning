#!/usr/bin/env python3
"""Module contains function for
creating a pandas dataframe from
a numpy array."""


import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Args:
        array: np.ndarray from which you should create the pd.DataFrame.

    Return: pd.DataFrame
    """

    col = [chr(i) for i in range(ord('A'),ord('A')+array.shape[1])]
    
    return pd.DataFrame(data=array, columns=col)
