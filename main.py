import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# %matplotlib inline

def mc(data, predict_days, trials = 1000):
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
    # Z = np.zeros_like(np.random.rand(days,trials))
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

def reg(theta1, theta2, total_days, predict_days):
    reg_predict = []
    for i in range(total_days - predict_days, total_days):
        reg_predict.append(theta1 * i + theta2)
    return reg_predict

def mse(prediction, real):
    if len(prediction) != len(real):
        print(len(prediction))
        print(len(real))
        return -1
    length = len(prediction)
    mse = 0
    for i in range(length):
        error = prediction[i] - real[i]
        mse = mse + (error**2)
    mse = mse / length
    return mse

def main(stock, theta1 = 0.5, theta2 = 200, reg = False):
    stock_file = stock + ".csv"
    raw = pd.read_csv(stock_file)
    raw = raw[::-1].reset_index(drop = True)    # Reset index, start with the oldest record
    data = raw.loc[:," Close"]
    data.columns = ["Close"]
    total_days = data.size
    predict_days = 200
    real_data = data.iloc[-predict_days:].reset_index(drop = True)  # extract the last 200 days for the comparison
    training_data = data.iloc[0:total_days - predict_days].reset_index(drop = True) 
    training_data = pd.DataFrame(training_data)

    trials = 1000
    theta1 = 0.5195
    theta2 = 258.2598

    if reg == True:
        reg_price = reg(theta1, theta2, total_days, predict_days)
        reg_mse = mse(reg_price, real_data)

    mc_model, mc_price = mc(training_data, predict_days, trials)

    mc_mse = mse(mc_price, real_data)

    fig, ax = plt.subplots()
    ax.set_title("prediction over 200 days of " + stock)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.plot(real_data, label = "Real stock price")
    ax.plot(mc_price, label = "Monte Carlo")
    if reg == True:
        ax.plot(reg_price, label = "Linear Regression")
    ax.legend()

    fig2, ax2 = plt.subplots()
    ax2.set_title("10 example of Monte Carlo of " + stock)
    ax2.plot(mc_model[:,0:10])

    print("stat of " + stock)

    print("mse of monte carlo:")
    print(mc_mse)
    if reg == True:
        print("mse of linear regression:")
        print(reg_mse)

    plt.show()

stock = "BAC"
slope = 0.5195
intercept = 258.2598
main(stock, slope, intercept)