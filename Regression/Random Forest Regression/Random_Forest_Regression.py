#%%
# Random Forest Regression
# An example of Ensemble Learning, applying one algorithm several times or
# taking in more than one algorithm to make the model more powerful.

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
# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
RandomForest = RandomForestRegressor(n_estimators = 10 , random_state = 0)
RandomForest.fit(X, y)

#%%
# Making Predictions
print(RandomForest.predict([[value]]))

#%%
# Smooth Visualization of Random Forest Regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, RandomForest.predict(X_grid), color = 'blue')
plt.title("Graph Name")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()