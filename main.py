import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
# %matplotlib inline

def mc(data, predict_days, trials = 1000):
    print(data)
    print(data.size)
    pct_change = []
    for i in range(data.size):
        if i == data.size - 1:
            break
        new = data.iloc[i + 1]
        old = data.iloc[i]
        pct_change.append((new - old) / old)

    pct_change = pd.DataFrame(pct_change)
    log_returns = np.log(1 + pct_change)

    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5*var)
    stddev = log_returns.std()

    days = predict_days + 1
    Z = norm.ppf(np.random.rand(days,trials)) # Percent point function
    daily_returns = np.exp(drift.values + stddev.values * Z)

    S0 = data.iloc[-1]
    print(S0)
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    avg_prediction = []

    for t in range(1,days):
       price_list[t] = price_list[t-1]*daily_returns[t]
       avg_prediction.append(price_list[t].mean())

    return price_list, avg_prediction

def reg(data):
    
    moded_x = PolynomialFeatures(degree = 1, include_bias=False).fit_transform(x)
    model = LinearRegression().fit(moded_x, y)
    err = model.score(moded_x,y)
    intercept = model.intercept_
    result = model.predict(moded_x)
    return result, intercept, err

def mse(prediction, real):
    if len(prediction) != len(real):
        return -1
    length = len(prediction)
    mse = 0
    for i in range(length):
        error = prediction[i] - real[i]
        mse = mse + (error**2)

    mse = mse / length

    return mse

raw = pd.read_csv("AMD.csv")
raw = raw[::-1].reset_index(drop = True)    # Reset index, start with the oldest record
data = raw.loc[:," Close"]
data.columns = ["Close"]
total_days = data.size
predict_days = 200
real_data = data.iloc[-predict_days:].reset_index(drop = True)  # extract the last 200 days for the comparison
training_data = data.iloc[0:total_days - predict_days].reset_index(drop = True) 
training_data = pd.DataFrame(training_data)

trials = 1000
mc_model, mc_price = mc(training_data, predict_days, trials)
# reg_price = reg(training_data)

mc_mse = mse(mc_price, real_data)
# reg_mse = mse(reg_price, real_data)

fig, ax = plt.subplots()
ax.set_title("prediction over 200 days")
ax.set_xlabel("date")
ax.set_ylabel("price")
ax.plot(real_data, label = "Real stock price")
ax.plot(mc_price, label = "Monte Carlo")
# ax.plot(reg_price, label = "Linear Regression")
ax.legend()

fig2, ax2 = plt.subplots()
ax2.plot(mc_model)
print("mse of monte carlo:")
print(mc_mse)
# print("mse of linear regression:")
# print(reg_mse)

plt.show()
