import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew



df = pd.read_csv(r'C:\Users\Sebastian Sjöholm\Downloads\AAPL_.csv', engine='python', parse_dates=['Date'])
df = df.set_index('Date').sort_index(ascending=False)

np_dly_prices = np.array(df["Close"][df.index >= '2020-01-01'].dropna())
np_dly_change = np.log(np.delete(np_dly_prices, -1)) - np.log(np.delete(np_dly_prices, 0))
print("Log normal stat data: ","MAX: " + str(max(np_dly_change)), "MIN: "+str(min(np_dly_change)), "MEAN: "+str(np.mean(np_dly_change)), "STD: " +str(np.std(np_dly_change)))
print('The 1-day 99.5 Historical VaR in pct terms (assuming LogN prices) is: ' + str(np.percentile(np_dly_change, q=0.5, interpolation='lower')))





