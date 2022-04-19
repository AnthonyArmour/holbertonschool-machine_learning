#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(columns='Weighted_Price', inplace=True)
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = df["Date"].map(datetime.fromtimestamp)
df.set_index('Date', inplace=True)
df = df[df.index.year >= 2017]

df['Close'].fillna(method='ffill', inplace=True)
vals = {
    "High": df.Close, "Low": df.Close, "Open": df.Close,
    "Volume_(BTC)": 0, "Volume_(Currency)": 0}
df.fillna(value=vals, inplace=True)

mp = {"High": "max", "Low": "min",
        "Close": "last", "Open": "mean",
        "Volume_(BTC)": "sum", "Volume_(Currency)": "sum"}

df = df.resample('D').agg(mp)
plt.rcParams["axes.formatter.limits"] = (0, 35000000)
axes = df.plot(figsize=(20, 10))
plt.show()