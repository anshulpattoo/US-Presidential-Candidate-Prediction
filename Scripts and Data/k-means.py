import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

dataset = pd.read_csv('mergednowinners.csv')
X = dataset.iloc[:, 1:].values


kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=10000, n_init=10, random_state=None)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()