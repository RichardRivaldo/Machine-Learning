#%%
# K-Nearest Neighbors (K-NN)

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
# Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#%% Check
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
# KNN Model
# Optimal N Neighbors can be determined using Elbow Theorem
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = X, metric = 'minkowski', p = 2)
KNN.fit(X_train, y_train)

#%%
# Making Predictions on Training Set
print(KNN.predict(sc.transform([[value1,value2]])))

#%%
# Making Predictions on Test Set
# Comparing the prediction with the y_test set
Prediction = KNN.predict(X_test)
print(np.concatenate((Prediction.reshape(len(Prediction),1), y_test.reshape(len(y_test),1)),1))

#%%
# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ConfusionMatrix = confusion_matrix(y_test, Prediction)
print(ConfusionMatrix)

#%% 
#Accuracy
Accuracy = accuracy_score(y_test, Prediction)
print(Accuracy)

#%%
# Visualization of Training Set
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, KNN.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Graph Name')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend()
plt.show()

#%%
# Visualization of Test Set
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, KNN.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Graph Name')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend()
plt.show()