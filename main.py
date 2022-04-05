import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import monte_carlo as mc
# import regression as reg
# %matplotlib inline

raw = pd.DataFrame()
raw = pd.read_csv("AMD.csv")
raw[:10]
raw.set_index(pd.DatetimeIndex(raw['Date']), inplace=True)
data = raw.loc[:," Close"]
data.columns = ["Close"]
data = pd.DataFrame(data)

mc_price = mc.mc(data)
# reg_price, reg_mse = reg.reg(data)

fig, ax = plt.subplots()
ax.set_title("prediction over 200 days")
ax.set_xlabel("date")
ax.set_ylabel("price")
ax.plot(data, label = "Real stock price")
ax.plot(mc_price, label = "Monte Carlo")
# ax.plot(reg_price, label = "Linear Regression")