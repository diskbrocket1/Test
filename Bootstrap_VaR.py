
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

# Here we will calculate boostrap VaR. This is done by taking using the past N days of price changes and
# sample randomly WITH replacement (this means you can pick the same sample element several times) creating a sample of
# N elements from the original sample of N (which is the past N price changes). The sampling with replacement is done
# several times, mabey 10000, and then the VaR is calculated for each sample. Then take the mean and you are done


VaR_days = 252
simuations = 10000

df = pd.read_csv(r'C:\Users\Sebastian Sjöholm\Downloads\AAPL_.csv', engine='python', parse_dates=['Date'])
df = df.set_index('Date').sort_index(ascending=False)
dly_diff = df['Close'][df.index >= '2020-01-01'].diff(periods=-1).dropna()
np_prices = np.array(df['Close'])[:VaR_days]
np_dly_cng = np.log(np.delete(np_prices, -1)) - np.log(np.delete(np_prices, 0))



########Funtion for calculating historic VaR from a sample########
def var_995(sample):
    historic_VaR = np.percentile(sample , q=0.5, interpolation = 'lower')
    return historic_VaR


########draiwing random sample (with replacement) of N elements from data set of N elements########
def random(data):
    random_sample = np.random.choice(data, len(data))
    return random_sample

########calcilating a np array of X historic VaR########
def var_sampling(data, size):
    result = np.empty(size)

    for i in range(size):
        x = random(data)
        result[i] = var_995(x)

    return result


########Now we call the actual calculation########

Bootstrap_var = var_sampling(np_dly_cng, simuations)

print(f'The Bootstrap Historical VaR is: {np.mean(Bootstrap_var)}')
