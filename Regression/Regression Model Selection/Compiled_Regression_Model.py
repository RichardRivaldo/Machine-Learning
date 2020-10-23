#%%
# Compiled Regression Model
# Model Selection for Regression

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Dataset
dataset = pd.read_csv("Data.csv")

#%%
# Features and Dependent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%%
# Splitting The Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#%%
# Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%%
# Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%%
# Polynomial Regression Model
# Matrix of Variables
from sklearn.preprocessing import PolynomialFeatures
PolynomialClass = PolynomialFeatures(degree = n)        # Degrees for better model, might cause overfitting
X_Poly = PolynomialClass.fit_transform(X)

# Polynomial Regression Model
regressor = LinearRegression()
regressor.fit(X_Poly, y)

#%%
# SVR Model
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")                     # Radial Basis Function Kernel
regressor.fit(X, y)

#%%
# Decision Tree Regression Model
from sklearn.tree import  DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#%%
# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = n , random_state = 0)
regressor.fit(X, y)

#%%
# These models will need some preprocessing first to suit the data needed by each models

#%%
# Making Predictions
# Comparing the actual results with the predicted results
Prediction = regressor.predict(X_test)
np.set_printoptions(precision = 3)
print(np.concatenate((Prediction.reshape(len(Prediction), 1), y_test.reshape(len(y_test), 1)), 1))

#%%
# Evaluating Model Performance Function
# Using Metrics Module for Regression (R-Squared)
def Evaluate(y_test, Prediction):
    from sklearn.metrics import r2_score
    r2_score(y_test, Prediction)