import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew
import yfinance as yf
from yahoofinancials import YahooFinancials

tsla_df = yf.download(['TSLA','AAPL'],
                      start='2019-01-01',
                      end='2019-12-31',
                      progress=False)

print(tsla_df.info())
print(tsla_df.columns)


plt.plot(tsla_df.index.to_pydatetime(), tsla_df['Close'])

plt.show()