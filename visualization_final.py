'''This program plots the visualization for points of closest approach observed in Muon TOMOgraphy.2D and 3D Visualization are created for both raw and filtered dataset in order to gain better analysis of the impurity in raw dataset of points and to observe outliers'''

#Import pandas package for creating Dataframes
import pandas as pd
#Import matplotlib and mpl_toolkits for visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 6})

#creating dataframe from the .csv file
#Function for reading the csv file
def readfile(filename):
	DF=pd.read_csv(filename)
	return DF

#Initializing ploting space
fig=plt.figure()


#Function for making 2D plots 
'''Two axes will be used for plotting the graph so the parameters are axis1,axis2.Here we are assuming that 4 graphs need to be plotted so we are dividing plot space into 4 subdivisions i.e. 2 rows and 2 columns'''
def twodimensional_plot(axis1,axis2):
	if(twodimensional_plot.counter>2):
		print("Max plotting limit reached.....")
	else:
		ax1=fig.add_subplot(2,2,twodimensional_plot.counter)
		ax1.scatter(axis1,axis2,s=2)
		'''ax1.set_xlabel('X-COORDINATE OF RAW POCA')
		ax1.set_ylabel('Y-COORDINATE OF RAW POCA')
		ax1.set_title('2D VISUALIZATION OF RAW POCA')'''
		twodimensional_plot.counter+=1
twodimensional_plot.counter=1
raw_dataframe=readfile('raw.csv')
filtered_dataframe=readfile('filtered.csv')

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
twodimensional_plot(xaxis_raw,yaxis_raw)
twodimensional_plot(xaxis_filtered,yaxis_filtered)

#Display the results
plt.show()


