#!/usr/bin/env python3
"""Module contains function for
creating a pandas dataframe from
a csv file."""


import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame.

    Args:
        filename: The file to load from.
        delimiter: The column separator.

    Return: loaded pd.DataFrame
    """

    return pd.read_csv(filename, delimiter=delimiter)
