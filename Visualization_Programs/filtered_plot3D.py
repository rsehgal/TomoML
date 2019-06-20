#Import pandas package for creating Dataframes
import pandas as pd

#Import matplotlib and mpl_toolkits for visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#creating dataframe from the .csv file
raw_dataframe=pd.read_csv('filtered.csv')

#Selecting series out of the Dataframe i.e. y-coordinate of poca
yaxis=raw_dataframe['Y']

#Selecting series out of the Dataframe i.e. x-xoordinate of poca
xaxis=raw_dataframe['X']
print(xaxis)


#Selecting series out of the Dataframe i.e. z-coordinate of poca
zaxis=raw_dataframe['Z']

#Visualization
#colors=['red','green','blue']
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xaxis,yaxis,zaxis,zdir='x')
plt.show()
