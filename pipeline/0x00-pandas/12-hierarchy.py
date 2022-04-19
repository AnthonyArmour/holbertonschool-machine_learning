#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1.set_index('Timestamp', inplace=True)
df2.set_index('Timestamp', inplace=True)

frames = [df2.loc[: 1417411920], df1]

df = pd.concat(frames, axis=0, keys=["bitstamp", "coinbase"])

df = df.reorder_levels([1, 0], axis=0)
df.sort_index(inplace=True)

print(df)
