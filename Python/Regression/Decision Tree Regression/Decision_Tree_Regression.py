#%%
# Decision Tree Regression

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
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#%%
# Decision Tree Regression Model
from sklearn.tree import  DecisionTreeRegressor
DTreeRegressor = DecisionTreeRegressor(random_state = 0)
DTreeRegressor.fit(X, y)

#%%
# Making Predictions
print(DTreeRegressor.predict([[value]]))

#%% 
# Smooth Visualization of Decision Tree Regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, DTreeRegressor.predict(X_grid), color = 'blue')
plt.title("Graph Name")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()

#%%
# Notes
# Decision Tree Regression is not a good choice here because the data is only in 1D
# and the chance of overfitting to happen is so high as can be seen from the graph
# Decision Tree Regression will work very well in high dimension datasets