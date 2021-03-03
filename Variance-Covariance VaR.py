import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew
import yfinance as yf
from yahoofinancials import YahooFinancials


w = np.array([0.5, 0.5])

df = yf.download(['TSLA','AAPL'],
                      start='2019-01-01',
                      end='2019-12-31',
                      progress=False)


df_close = df[[("Close","AAPL"),("Close","TSLA")]].dropna()

df_close = df_close.to_numpy().T

df_close = np.log(np.delete(df_close, -1, 1)) - np.log(np.delete(df_close, 0, 1))

cov_mat = np.cov(df_close, bias=False)

avg_rates =  np.mean(df_close, axis=1).dot(w.T)

print(avg_rates)

port_std = np.sqrt(w.dot(cov_mat).dot(w.T))*-1.645

print(port_std)







# print(np.cov(df_close, bias=False))
# print(np.mean(df_close[0][:]))
# print(np.mean(df_close[1][:]))













# plt.plot(tsla_df.index.to_pydatetime(), tsla_df['Close'])
#
# plt.show()