import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
# %matplotlib inline

raw = pd.DataFrame()
raw = pd.read_csv("AMD.csv")
raw[:10]