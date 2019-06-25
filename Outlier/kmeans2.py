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


#Storing dataset in dataframe data structure of pandas
#poca_dataframe=pd.read_csv('../dataset/filtered.csv')
poca_dataframe=readfile('../dataset/filteredWithDoCA.csv')
corr=poca_dataframe.corr(method='pearson')

#Extracting individual series of attributes from the whole dataframe

zaxis=poca_dataframe['Z']
scat_angle=poca_dataframe['Scat_Angle']
doca=poca_dataframe['doca']
print(poca_dataframe)
info_filtered_With_DoCA=poca_dataframe.describe()
print(info_filtered_With_DoCA)


xlabel="X-AXIS OF FILTERED POCA"
ylabel="Y-AXIS OF FILTERED POCA"
zlabel="Z- AXIS OF FILTERED POCA"
sorted_docalabel="Sorted Doca"
docalabel="Distance Of Closest Approach(DocA)"
SIZE_label="SIZE of Cluster"
log_of_doca_label="log(doca)"
distance_label="CEntroid distance"
log_of_CEntroid_DIstance="log(centroid_distancE)"
log_of_sorted_doca_label="log(doca)(sorted)"


mean_scat_angle=info_filtered_With_DoCA.loc['mean']['Scat_Angle']
std_scat_angle=info_filtered_With_DoCA.loc['std']['Scat_Angle']
limit1=mean_scat_angle-2*std_scat_angle
limit2=mean_scat_angle+2*std_scat_angle
poca_dataframe=poca_dataframe[poca_dataframe['Scat_Angle']>=limit1]
poca_dataframe=poca_dataframe[poca_dataframe['Scat_Angle']<=limit2]

xaxis=poca_dataframe['X']
yaxis=poca_dataframe['Y']
twodimensional_plot(xaxis,yaxis)
display()

