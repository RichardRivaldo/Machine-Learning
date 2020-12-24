#%%
# Polynomial Regression

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Dataset
dataset = pd.read_csv("")

#%%
# Features and Dependent Variables
X = dataset.iloc[:, 1:-1].values            # Only needs the second column and no need encoding
y = dataset.iloc[:, -1].values

#%%
# Splitting The Dataset
# The sample dataset is small in term of size so splitting the dataset is not needed anymore

#%%
# Linear Regression Model
from sklearn.linear_model import LinearRegression
LinearRegressor = LinearRegression()
LinearRegressor.fit(X, y)

#%%
# Matrix of Variables
from sklearn.preprocessing import PolynomialFeatures
PolynomialClass = PolynomialFeatures(degree = n)        # Degrees for better model, might cause overfitting
X_Poly = PolynomialClass.fit_transform(X)

# Polynomial Regression Model
PolyReg = LinearRegression()
PolyReg.fit(X_Poly, y)

#%%
# Linear Regression Model Visualization
plt.scatter(X, y, color = "red")
plt.plot(X, LinearRegressor.predict(X), color = "blue")
plt.title("Graph Title")
plt.xlabel("X Label")
plt.ylabel(" Y Label")
plt.show()

#%%
# Polynomial Regression Model Visualization
plt.scatter(X, y, color = "red")
plt.plot(X, PolyReg.predict(X_Poly), color = "blue")
plt.title("Graph Title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()

#%%
# Smoother Polynomial Visualization
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, PolyReg.predict(PolynomialClass.fit_transform(X_grid)), color = "blue")
plt.title("Graph Title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()

#%%
# Making Predictions with Linear Regression Model
print(LinearRegressor.predict([[value]]))

#%%
# Making Predictions with Polynomial Regression Model
print(PolyReg.predict(PolynomialClass.fit_transform([[value]])))