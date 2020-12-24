#%%
# Multiple Linear Regression
# No need Feature Scaling

#%%
# Notes About Multiple Linear Regression

# There are mathematical assumptions that we need to check about the dataset so that 
# it fulfills the characteristic of linear regression
# Important Statistical Intuition: Null Hypothesis, P-Values, and Significant Level 

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Dataset
dataset = pd.read_csv("")

#%%
# Feature and Dependent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%%
# Check
print(X)

#%%
# Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#%%
# Check
print(X)

#%%
# Splitting The Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#%%
# Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
MultiRegressor = LinearRegression()
MultiRegressor.fit(X_train, y_train)

#%%
# Making Prediction
Prediction = MultiRegressor.predict(X_test)
np.set_printoptions(precision = 2)

# Comparing Predictions With The Test Set By Concatenate Both Reshaped Vectors
print(np.concatenate((Prediction.reshape(len(Prediction), 1), y_test.reshape(len(y_test), 1)), 1))

#%%
# Single Data Prediction
# Basically, fill all features (including encoded one) of prediction data in a 2D Array
# Predict receives 2D Array
Prediction = MultiRegressor.predict([[value1, value2, value3]])
print(Prediction)

#%%
# Regression Equation
# Will include dummy variables too
Coefficient = MultiRegressor.coef_
Intercept = MultiRegressor.intercept_

#%%
# Visualization is not really good for Multiple Linear Regression
# Can be done in 3D Visualization