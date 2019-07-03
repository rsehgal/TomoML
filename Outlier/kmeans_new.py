'''This program is about the implementation of the K-Means Clustering Algorithm on the poca dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''
import sys
from math import *
sys.path.append("/home/TomoML/'Visualization Programs'/")
#Import sklearn for implementing DBSCAN and suitable data normalization
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
from visualization_funcs import *

#Import pandas and numpy for data accessing and manipulation
import numpy as np
import pandas as pd


def kmeans(no_of_clusters,dataset):

	#Storing dataset in dataframe data structure of pandas
	#poca_dataframe=pd.read_csv('../dataset/filtered.csv')
	poca_dataframe=readfile(dataset)
	corr=poca_dataframe.corr(method='pearson')
	


	#COnversion toa NUMPY ARRaY
	poca_dataframe_numpy = poca_dataframe.as_matrix().astype("float32", copy = False)



	#Clustering
	kmeans=KMeans(n_clusters=no_of_clusters,precompute_distances='auto')
	kmeans.fit(poca_dataframe_numpy[:,:])
	labels=kmeans.labels_

	


	#Total no of clusters
	print(labels)
	print(len(labels))
	poca_dataframe['Labels']=labels
	
	

		
	'''twodimensional_plot(labels,xaxis,yaxis,xlabel,ylabel)
	twodimensional_plot(labels,yaxis,zaxis,ylabel,zlabel)
	twodimensional_plot(labels,xaxis,zaxis,xlabel,zlabel)'''


	#Finding 5point summary of individual clusters
	cluster1_df=poca_dataframe[poca_dataframe['Labels']==0.0]
	cluster2_df=poca_dataframe[poca_dataframe['Labels']==1.0]
	cluster3_df=poca_dataframe[poca_dataframe['Labels']==2.0]
	cluster4_df=poca_dataframe[poca_dataframe['Labels']==3.0]
	final_clusters=[]
	for i in range(no_of_clusters):
		cluster_df=poca_dataframe[poca_dataframe['Labels']==i]
		final_clusters.append(cluster_df)
	return final_clusters

