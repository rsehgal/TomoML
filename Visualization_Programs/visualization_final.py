'''This program plots the visualization for points of closest approach observed in Muon TOMOgraphy.2D and 3D Visualization are created for both raw and filtered dataset in order to gain better analysis of the impurity in raw dataset of points and to observe outliers'''

#Import the visualization_funcs module to import all necessary functions and packages for visualization
from visualization_funcs import *

#Import the datasets for visualization
raw_dataframe=readfile('../dataset/raw.csv')
filtered_dataframe=readfile('../dataset/filtered.csv')

#Selecting series out of the Dataframe i.e. x-xoordinate of poca
xaxis_raw=raw_dataframe['X']
xaxis_filtered=filtered_dataframe['X']

#print(xaxis)

#Selecting series out of the Dataframe i.e. y-coordinate of poca
yaxis_raw=raw_dataframe['Y']
yaxis_filtered=filtered_dataframe['Y']

#Selecting series out of Dataframe i.e. z-coordinate of poca
zaxis_raw=raw_dataframe['Z']
zaxis_filtered=filtered_dataframe['Z']
	
'''#Dummy variables to store the number of calls to the functions 
count_2D=1
count_3D=1'''
#Now we will plot two plots 2D plots and 3D plots namely
raw_xlabel='X-COORDINATE OF RAW POCA'
raw_ylabel='Y-COORDINATE OF RAW POCA'
raw_zlabel='Z-COORDINATE OF RAW POCA'

filtered_xlabel='X-COORDINATE OF FILTERED POCA'
filtered_ylabel='Y-COORDINATE OF FILTERED POCA'
filtered_zlabel='Z-COORDINATE OF FILTERED POCA'

twodimensional_plot(xaxis_raw,yaxis_raw,raw_xlabel,raw_ylabel)
twodimensional_plot(xaxis_filtered,yaxis_filtered,filtered_xlabel,filtered_ylabel)

threedimensional_plot(xaxis_raw,yaxis_raw,zaxis_raw)
threedimensional_plot(xaxis_filtered,yaxis_filtered,zaxis_filtered,filtered_xlabel,filtered_ylabel,filtered_zlabel)

display()


