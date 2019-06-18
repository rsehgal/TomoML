'''This program is about the implementation of the DBSCAN i.e. Density-based Spatial Clustering of Applications with Noise on the raw dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''

#Import sklearn for implementing DBSCAN and suitable data normalization
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import MinMaxScaler

#Import pandas and numpy for data accessing and manipulation
import numpy as np
import pandas as pd

#IMport Matplotlib for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Storing dataset in dataframe data structure of pandas
#raw_dataframe=pd.read_csv('../dataset/filtered.csv')
raw_dataframe=pd.read_csv('../dataset/raw.csv')
#Extracting individual series of attributes from the whole dataframe
xaxis=raw_dataframe['X']
yaxis=raw_dataframe['Y']
zaxis=raw_dataframe['Z']
scat_angel=raw_dataframe['scatteringAngle']


#Visualization

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
colorbar=ax.scatter(xaxis,yaxis,zaxis,c=scat_angel,cmap=plt.magma(),s=2)
fig.colorbar(colorbar)
#plt.show()


#Normalization
raw_dataframe = raw_dataframe.as_matrix().astype("float32", copy = False)
raw_dataframe=raw_dataframe[:,:-1]
stscaler=MinMaxScaler(feature_range=(0,1))
raw_dataframe = stscaler.fit_transform(raw_dataframe)
r=pd.DataFrame(raw_dataframe)
col_names=['X','Y','Z']
r.names=col_names
print(r.describe())
print(type(raw_dataframe))

print(raw_dataframe)
dbsc = DBSCAN(eps=0.52,metric='euclidean',min_samples=1500)
dbsc.fit(raw_dataframe)
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True
#Total no of clusters
print(labels)
print(len(labels))

counter_0=0
counter_1=0
for i in labels:
  if(i==0):
    counter_0=counter_0+1
  else:
    counter_1=counter_1+1			

print("Counter_0 : ",counter_0)
print("Counter_1 : ",counter_1)
print("TOtal no of clusters=",len(set(labels)))
#print(labels[labels>=0].sum())

