import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import DataFrame 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from plotnine import *
 
X = pd.read_csv('mostfreq1000docword.csv')

X.head()

EM = GaussianMixture(n_components = 2) 
EM.fit(X)

cluster = EM.predict(X)
print((cluster == 0).sum())
