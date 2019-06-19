'''This program is about the implementation of the DBSCAN i.e. Density-based Spatial Clustering of Applications with Noise on the raw dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''
import sys
print(sys.path)
sys.path.append("/home/TomoML/Outlier/")
#Import sklearn for implementing DBSCAN and suitable data normalization
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import MinMaxScaler
#from visualization_funcs import *
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
'''ax=fig.add_subplot(2,2,1,projection='3d')
colorbar=ax.scatter(xaxis,yaxis,zaxis,c=scat_angel,cmap=plt.magma(),s=2)
fig.colorbar(colorbar)'''


#Normalization
raw_dataframe = raw_dataframe.as_matrix().astype("float32", copy = False)
raw_dataframe=raw_dataframe[:,:-1]
stscaler=MinMaxScaler(feature_range=(0,1))
raw_dataframe = stscaler.fit_transform(raw_dataframe)
col_names1=['X','Y','Z']
raw_fr=pd.DataFrame(raw_dataframe,columns=col_names1)
print(raw_fr.describe())
print(type(raw_dataframe))

print(raw_dataframe)
dbsc = DBSCAN(eps=0.3,metric='manhattan',min_samples=9870)
dbsc.fit(raw_dataframe)
labels = dbsc.labels_

#Total no of clusters
print(labels)
print(len(labels))
final=np.column_stack((raw_dataframe,labels))
plot=final[final[:,3]==0]
print(plot)
col_names2=['X','Y','Z','Labels']
print(col_names2)
final_df=pd.DataFrame(plot,columns=col_names2)
print(final_df.describe())	
ax1=fig.add_subplot(1,1,1,projection='3d')
ax1.scatter(final_df['X'],final_df['Y'],final_df['Z'],zdir='z',s=2)

print("TOtal no of clusters=",len(set(labels)))
#print(labels[labels>=0].sum())
plt.show()


