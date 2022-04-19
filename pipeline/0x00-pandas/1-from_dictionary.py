#!/usr/bin/env python3
"""Module contains function for
creating a pandas dataframe from
a dictionary."""


import pandas as pd


df = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}
df = pd.DataFrame(df, index=['A', 'B', 'C', 'D'])
