import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
# %matplotlib inline

def mc(data):
    data.reindex(index=data.index[::-1])
    pct_change = []
    for i in range(data.size):
        if i == data.size - 1:
            break
        new = data.iloc[i]
        old = data.iloc[i+1]
        pct_change.append((new - old) / old)

    pct_change = pd.DataFrame(pct_change)
    log_returns = np.log(1 + pct_change)

    u = log_returns.mean()
    var = log_returns.var()

    drift = u - (0.5*var)
    stddev = log_returns.std()

    days = 200
    trials = 10000

    Z = norm.ppf(np.random.rand(days,trials)) # Percent point function

    daily_returns = np.exp(drift.values + stddev.values * Z)
    daily_returns
   
    S0 = data.iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    avg_prediction = []

    for t in range(1,days):
       price_list[t] = price_list[t-1]*daily_returns[t]

    return price_list

def reg(data):
    
    moded_x = PolynomialFeatures(degree = 1, include_bias=False).fit_transform(x)
    model = LinearRegression().fit(moded_x, y)
    err = model.score(moded_x,y)
    intercept = model.intercept_
    result = model.predict(moded_x)
    return result, intercept, err

def mse(prediction, real):
    if len(prediction) != len(real):
        return undefined
    
    length = len(prediction)
    mse = 0

    for i in length:
        error = prediction,iloc[i] - real.iloc[i]
        mse = mse + (error**2)

    mse = mse / length

    return mse
        
raw = pd.read_csv("AMD.csv")
# raw[:10]
raw.set_index(pd.DatetimeIndex(raw['Date']), inplace=True)
data = raw.loc[:," Close"]
data.columns = ["Close"]
data = pd.DataFrame(data)

mc_price = mc(data)
# reg_price = reg(data)

mc_mse = mse(mc_price, data)
# reg_mse = mse(reg_price, data)

    


fig, ax = plt.subplots()
ax.set_title("prediction over 200 days")
ax.set_xlabel("date")
ax.set_ylabel("price")
ax.plot(data, label = "Real stock price")
ax.plot(mc_price, label = "Monte Carlo")
# ax.plot(reg_price, label = "Linear Regression")

plt.show()