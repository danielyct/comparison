import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
# matplotlib inline
raw = pd.DataFrame()
raw = pd.read_csv("AMD.csv")
raw[:10]
raw.set_index(pd.DatetimeIndex(raw['Date']), inplace=True)
data = raw.loc[:," Close"]
data.columns = ["Close"]
data = pd.DataFrame(data)

log_returns = np.log(1 + data.pct_change())

sns.distplot(log_returns.iloc[1:])
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
data.plot(figsize=(15,6))
log_returns.plot(figsize=(15,6))
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)
stddev = log_returns.std()
x = np.random.rand(10,2)
norm.ppf(x)
Z = norm.ppf(np.random.rand(50,10000))
t_intervals = 1000
iterations = 10
daily_returns = np.exp(drift.values + stddev.values * norm.ppf(np.random.rand(50,1000)))
daily_returns
S0 = data.iloc[-1]
price_list = np.zeros_like(daily_returns)

price_list[0] = S0
price_list
for t in range(1,50):
    price_list[t] = price_list[t-1]*daily_returns[t]

plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(price_list).iloc[:,0:10])
sns.distplot(pd.DataFrame(price_list).iloc[-1])
plt.xlabel("Price after 50 days")
df = pd.DataFrame(price_list)

for i in range(len(df.columns)):
    sns.distplot(df[i])
      
