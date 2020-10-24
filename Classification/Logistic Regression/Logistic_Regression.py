#%%
# Logistic Regression
# Use sigmoid function

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
# Splitting The Dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#%%
# Check
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# Check
print(X_train)
print(X_test)

#%%
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
LogisticClassifier = LogisticRegression(random_state = 42)
LogisticClassifier.fit(X_train, y_train)

#%%
# Making Predictions on Training Set
print(LogisticClassifier.predict(sc.transform([[value1, value2, value3]])))

#%%
# Making Predictions on Test Set
# Comparing the prediction with the y_test set
Prediction = LogisticClassifier.predict(X_test)
print(np.concatenate((Prediction.reshape(len(Prediction), 1), y_test.reshape(len(y_test), 1)), 1))

#%%
# Confusion Matrix
# Column 
from sklearn.metrics import confusion_matrix, accuracy_score
ConfusionMatrix = confusion_matrix(y_test, Prediction)
print(ConfusionMatrix)

#%%
# Accuracy
Accuracy = accuracy_score(y_test, Prediction)
print(Accuracy)

#%%
# Visualization
# Only possible for 2 Features
# Correct Prediction: Dot Color = Region Color
# Logistic Regression is Linear Classifier and we can see this from the dividing
# line of the two regions in the visualization

#%%
# Visualization of Training Set
from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, LogisticClassifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title("Graph Name")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

#%%
# Visualization of Test Set
from matplotlib.colors import ListedColormap

X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, LogisticClassifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title("Graph Name")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

