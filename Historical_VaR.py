
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

VaR_days = 252

df = pd.read_csv(r'C:\Users\Sebastian Sjöholm\Downloads\AAPL_.csv', engine='python', parse_dates=['Date'])
df = df.set_index('Date').sort_index(ascending=False)
dly_diff = df['Close'][df.index >= '2020-01-01'].diff(periods=-1).dropna()
np_prices = np.array(df['Close'])[:VaR_days]


#########################################Using Normal prices#####################################################################
dly_pct_cng = (np.delete(np_prices, -1) - np.delete(np_prices, 0)) / np.delete(np_prices, 0)
dly_pct_cng = dly_pct_cng[~np.isnan(dly_pct_cng)]

historic_VaR_N = np.percentile(dly_pct_cng, q=0.5, interpolation='lower')

print(['Normal stat data: ', 'Mean: '+str(np.mean(dly_pct_cng)), 'Median: '+str(np.median(dly_pct_cng)), \
       'Std: '+str(np.std(dly_pct_cng)), 'Variance: '+str(np.var(dly_pct_cng)), 'Max: '+str(np.max(dly_pct_cng)), 'Min: ' +str(np.min(dly_pct_cng))])
print(f'The 1-day 99.5 Historical VaR in pct terms terms (using normal prices) is: {historic_VaR_N}')
print('Skew is: '+str(skew(dly_pct_cng)))
print('Kurtosis is: '+str(kurtosis(dly_pct_cng)))


#########################################Using Log N prices#####################################################################
LN_dly_change = np.log(np.delete(np_prices, -1)) - np.log(np.delete(np_prices, -0))
LN_dly_change = LN_dly_change[~np.isnan(LN_dly_change)]
historic_VaR_LN = np.percentile(LN_dly_change, q=0.5, interpolation='lower')
print("Log normal stat data: ","MAX: " + str(max(LN_dly_change)), "MIN: "+str(min(LN_dly_change)), \
      "MEAN: "+str(np.mean(LN_dly_change)), "STD: " +str(np.std(LN_dly_change)))
print(f'The 1-day 99.5 Historical VaR in pct terms (assuming LogN prices) is: {historic_VaR_LN}')

#########################################Plot#####################################################################


fig, axs = plt.subplots(2, 1)
axs[0].hist(dly_pct_cng, bins=100)
axs[0].set_xlabel('Based on Normal Prices' + '(Skew: '+str(round(skew(dly_pct_cng),5)) + \
                  '; Kurtosis: '+str(round(kurtosis(dly_pct_cng),5)) + "; 99.5% VaR: " \
                  +str(round(np.percentile(dly_pct_cng, q=0.5, interpolation='lower'),5))+ ")")
axs[0].set_ylabel('Frequency')
fig.suptitle("Dly pct changes based on Normal and LogNormal Prices", fontsize=25)



axs[1].hist(LN_dly_change, bins=100)
axs[1].set_xlabel('Based on LogNormal Prices' + '(Skew: '+str(round(skew(LN_dly_change),5)) + \
                  '; Kurtosis: '+str(round(kurtosis(LN_dly_change),5))+ "; 99.5% VaR: " \
                  + str(round(np.percentile(LN_dly_change, q=0.5, interpolation='lower'),5))+")")
axs[1].set_ylabel('Frequency')



plt.show()





