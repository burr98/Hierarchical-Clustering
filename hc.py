#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 

#get the dataset
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3, 4]].values

#use dendograms to find the number of clusters
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#fitting the hierarchical cluster to the data
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=10, c='blue', label = 'cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=10, c='red', label = 'cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=10, c='green', label = 'cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=10, c='cyan', label = 'cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=10, c='magenta', label = 'cluster 5')
plt.title('Hierarchical clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
