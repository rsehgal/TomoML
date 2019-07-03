'''This program is about the implementation of the K-Medoids Clustering Algorithm on the poca dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''
import sys
from math import *
sys.path.append("/home/TomoML/'Visualization Programs'/")
#Import sklearn for implementing K-Medoids
from sklearn.metrics.pairwise import pairwise_distances
import kmedoids
from visualization_funcs import *

#Import pandas and numpy for data accessing and manipulation
import numpy as np
import pandas as pd


#Storing dataset in dataframe data structure of pandas
#poca_dataframe=pd.read_csv('../dataset/filtered.csv')
poca_dataframe=readfile('../dataset/filteredWithDoCA.csv')
corr=poca_dataframe.corr(method='pearson')

#Extracting individual series of attributes from the whole dataframe
xaxis=poca_dataframe['X']
yaxis=poca_dataframe['Y']
zaxis=poca_dataframe['Z']
scat_angle=poca_dataframe['Scat_Angle']
doca=poca_dataframe['doca']
print(poca_dataframe)

#COnversion toa NUMPY ARRaY
poca_dataframe = poca_dataframe.as_matrix().astype("float32", copy = False)
# distance matrix
D = pairwise_distances(poca_dataframe, metric='euclidean')

# split into 4 clusters
M, C = kmedoids.kMedoids(D, 4)

print('medoids:')
for point_idx in M:
    print( data[point_idx] )

print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, data[point_idx]))
