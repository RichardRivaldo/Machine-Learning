#%%
# Hierarchical Clustering

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Dataset
dataset = pd.read_csv("")
X = dataset.iloc[:, [3, 4]].values

#%%
# Dendrogram -> Shows dissimilarity of points
# We can use this to find optimal threshold of the dendrogram
# Then determine the number of vertical lines crosses the threshold line
# We can also find the optimal threshold by finding the longest vertical line
# that doesn't cross any horizontal line and put the threshold on that vertical line
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Feature 1')
plt.ylabel('Euclidean Distances')
plt.show()

#%%
# Hierarchical Clustering Model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#%%
# Hierarchical Clusters Visualization
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Graph Name')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()