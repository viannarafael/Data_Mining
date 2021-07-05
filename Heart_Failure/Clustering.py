from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
import numpy as np

# Import dataset
# ------------------------------------------------------------------
df = pd.read_csv("dataset/heart_failure_clinical_records.csv",
                 sep=',', delimiter=None, header='infer')
X = df.drop(labels=['time', 'DEATH_EVENT'], axis=1)
print(X.describe())

# K-Means Clustering
# ------------------------------------------------------------------
km = KMeans(n_clusters=7)
km.fit(X)
y_label = pd.DataFrame(km.labels_)
df_KmeansCluster = pd.concat([X, y_label], axis=1)
# print(df_KmeansCluster)
print('\nMean\n')
print(df_KmeansCluster.groupby(by=0).mean())
print('\nStd\n')
print(df_KmeansCluster.groupby(by=0).std())
print('\nCount\n')
print(df_KmeansCluster.iloc[:, -1].value_counts())
