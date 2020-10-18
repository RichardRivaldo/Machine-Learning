#%%
# Simple Linear Regression
# No need Feature Scaling

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Dataset
data = pd.read_csv("")

#%%
# Features and Dependent Variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


#%%
# Splitting The Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#%%
# Linear Regression Model
from sklearn.linear_model import LinearRegression
LinearRegressor = LinearRegression()
LinearRegressor.fit(X_train, y_train)

#%%
# Making Prediction
Prediction_y = LinearRegressor.predict(X_test)

#%%
# Single Data Prediction
Prediction = LinearRegressor.predict([[value]])                             #Predict receives 2D Array
print(Prediction)

#%%
# Regression Equation
Coefficient = LinearRegressor.coef_
Intercept = LinearRegressor.intercept_
print("y = %fx + %f" %(Coefficient, Intercept))

#%%
# Training Set Visualization
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, LinearRegressor.predict(X_train), color = 'blue')
plt.title('Graphic Name')
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.show()

#%%
# Test Set Visualization
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, LinearRegressor.predict(X_train), color = 'blue')         #Same line equation with before
plt.title('Graphic Name')
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.show()