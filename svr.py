# Polynomial Regression

# Data prepprocessing
# Importing  Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

# Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X.reshape(-1, 1),y)