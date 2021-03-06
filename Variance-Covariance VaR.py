import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, norm
import yfinance as yf
from yahoofinancials import YahooFinancials



#Note that you must include more than one stock. --> should develope to work w/ one##
stocks = ["TSLA","GOOG", "AAPL", "AMZN", "FB"]

#stocks = ["GME", "AMC"]

InitialInvestment = 1000000
w = np.random.dirichlet(np.ones(len(stocks)),size=1)
var = 0.005
var_days = 10

df = yf.download(stocks,
                      start='2019-01-01',
                      end='2019-12-31',
                      progress=False)




df = df["Close"].dropna()

df = df.to_numpy()

df_n = (np.delete(df, -1, 0) - np.delete(df, 0, 0)) / np.delete(df, 0, 0)
df_log = np.log(np.delete(df, -1, 0)) - np.log(np.delete(df, 0, 0))


mean_n = np.mean(df_n, axis=0).dot(w.T)
mean_log = np.mean(df_log, axis=0).dot(w.T)

cov_mat_n = np.cov(df_n, rowvar=False, bias=False)
cov_mat_log = np.cov(df_log, rowvar=False, bias=False)

port_std_n = np.sqrt(w.dot(cov_mat_n).dot(w.T))
port_std_log = np.sqrt(w.dot(cov_mat_log).dot(w.T))

port_loss_log = norm.ppf(var, mean_log, port_std_log)
port_loss_n = norm.ppf(0.005, mean_n, port_std_n)



print("The weights are: " + str(w) + ", applied on the corresponding Stocks: " + str(stocks))
print("Assuming norm-Prices, you max loss using "+str(1-var) +" is: " + str(port_loss_n*InitialInvestment))
print("Assuming log-Prices, you max loss using "+str(1-var) +" is: " + str(port_loss_log*InitialInvestment))

var_array_n = np.empty(var_days)
var_array_log = np.empty(var_days)
for i in range(var_days):
    var_array_n[i] = norm.ppf(var, mean_n, port_std_n*np.sqrt(i+1))*InitialInvestment
    var_array_log[i] = norm.ppf(var, mean_log, port_std_log * np.sqrt(i + 1)) * InitialInvestment
    print("Your (Norm) " + str(i+1) + " day VaR is " + str(round(var_array_n[i],2)))


#########################################Plot#####################################################################


fig, axs = plt.subplots(2, 2)
axs[0,0].hist(df_n, bins=100)
x = np.linspace(mean_n - 4*port_std_n, mean_n + 4*port_std_n)
axs[0,0].plot(x, norm.pdf(x,mean_n, port_std_n))
axs[1,0].plot(var_array_n)


axs[0,1].hist(df_log, bins=100)
x = np.linspace(mean_log - 4*port_std_log, mean_log + 4*port_std_log)
axs[0,1].plot(x, norm.pdf(x,mean_log, port_std_log))
axs[1,1].plot(var_array_log)

plt.show()
