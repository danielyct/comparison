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
    
    days = predict_days + 1

    pct_change = pd.DataFrame(pct_change)
    log_returns = np.log(1 + pct_change)

    mean = log_returns.mean()
    var = log_returns.var()
    drift = mean - (0.5*var)
    stddev = log_returns.std()

    
    vol_table = norm.ppf(np.random.rand(days,trials)) # Percent point function
    # vol_table = np.zeros_like(np.random.rand(days,trials)) # Testing Monte Carlo without randomness/volatility
    daily_returns = np.exp(drift.values + stddev.values * vol_table) 

    S0 = data.iloc[-1]
    print(S0)   # debugging use
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    avg_prediction = []

    for t in range(1,days):     
       price_list[t] = price_list[t-1]*daily_returns[t]
       avg_prediction.append(price_list[t].mean())

    return price_list, avg_prediction

def reg(theta1, theta2, total_days, predict_days): # theta1 and theta2 is provided from the matlab code, input manually 
    reg_predict = []
    for i in range(total_days - predict_days, total_days):
        reg_predict.append(theta1 * i + theta2)
    return reg_predict

def mse(prediction, real):
    if len(prediction) != len(real):
        print(len(prediction))      # debugging use
        print(len(real))            # debugging use
        return -1
    length = len(prediction)
    mse = 0
    for i in range(length):
        error = prediction[i] - real[i]
        mse = mse + (error**2)
    mse = mse / length
    return mse

def main(stock, theta1 = 0.5, theta2 = 200, reg_on = False):
    stock_file = stock + ".csv"
    raw = pd.read_csv(stock_file)
    raw = raw[::-1].reset_index(drop = True)    # Reset index, start with the oldest record
    data = raw.loc[:,"Close"]
    data.columns = ["Close"]
    total_days = data.size
    predict_days = 50
    real_data = data.iloc[-predict_days:].reset_index(drop = True)  # extract the last 200 days for the comparison
    training_data = data.iloc[0:total_days - predict_days].reset_index(drop = True) 
    training_data = pd.DataFrame(training_data)

    trials = 1000

    mc_model, mc_price = mc(training_data, predict_days, trials)
    mc_mse = mse(mc_price, real_data)
    
    if reg_on == True:
        reg_price = reg(theta1, theta2, total_days, predict_days)
        reg_mse = mse(reg_price, real_data)

    fig, ax = plt.subplots()
    ax.set_title("prediction over "+ str(predict_days) +" days of " + stock)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.plot(real_data, label = "Real stock price")
    ax.plot(mc_price, label = "Monte Carlo")
    if reg_on == True:
        ax.plot(reg_price, label = "Linear Regression")
    ax.legend()

    fig2, ax2 = plt.subplots()
    ax2.set_title("Monte Carlo simulations of " + stock)
    ax2.plot(mc_model[:,:])

    print("stat of " + stock)

    print("mse of monte carlo:")
    print(mc_mse)
    if reg_on == True:
        print("mse of linear regression:")
        print(reg_mse)
    plt.show()
    
stock = "GOOGL"
slope = 3.047484420577
intercept = 1130.110241408717
main(stock, slope, intercept, reg_on = True)

# Ideally, we can run the below code with all thetas imported as a list
# stocks = ["AMD", "BAC", "GOOGL", "JPM", "MS", "TSLA"]
# slope = 0.007268825632074
# intercept = 0.365887407728254
# for stock in stocks:
#     main(stock, slope, intercept, reg_on = False)

