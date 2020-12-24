#%%
# Support Vector Regression (SVR)

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Dataset
dataset = pd.read_csv("")

#%%
# Features and Dependent Variable
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#%%
# Check

print(X)
print(y)

#%%
# Reshape y to Vertical 2D Array
y = y.reshape(len(y), 1)

#%%
# Check
print(y)

#%%
# Feature Scaling for Features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Feature Scaling for Dependent Variable
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#%%
# Check
print(X)
print(y)

#%%
# SVR Model
from sklearn.svm import SVR
SVR_Regressor = SVR(kernel = "rbf")                     # Radial Basis Function Kernel
SVR_Regressor.fit(X, y)

#%%
# Making Predictions
# Need inverse transform to get the real value of data (before we scaled it)
print(sc_y.inverse_transform(SVR_Regressor.predict(sc_X.transform([[value]]))))

#%%
# SVR Visualization
# Need inverse transform
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(SVR_Regressor.predict(X)), color = "blue")
plt.title("Graph Title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()

#%%
# Smoother SVR Visualization
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform((X))), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(X_grid, sc_y.inverse_transform(SVR_Regressor.predict(sc_X.transform(X_grid))), color = "blue")
plt.title("Graph Title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()