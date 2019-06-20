#Import pandas package for creating Dataframes
import pandas as pd
#Import matplotlib and mpl_toolkits for visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 6})
#creating dataframe from the .csv file
raw_dataframe=pd.read_csv('raw.csv')
filtered_dataframe=pd.read_csv('filtered.csv')

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

#Visualization
#colors=['red','green','blue']
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax1.scatter(xaxis_raw,yaxis_raw,s=2)
ax1.set_xlabel('X-COORDINATE OF RAW POCA')
ax1.set_ylabel('Y-COORDINATE OF RAW POCA')
ax1.set_title('2D VISUALIZATION OF RAW POCA')

ax2=fig.add_subplot(2,2,2)
ax2.scatter(xaxis_filtered,yaxis_filtered,s=2)
ax2.set_xlabel('X-COORDINATE OF FILTERED POCA')
ax2.set_ylabel('Y-COORDINATE OF FILTERED POCA')
ax2.set_title('2D VISUALIZATION OF FILTERED POCA')

ax3=fig.add_subplot(2,2,3,projection='3d')
ax3.scatter(xaxis_raw,yaxis_raw,zaxis_raw,zdir='z',s=2)
ax3.set_xlabel('X-COORDINATE OF RAW POCA')
ax3.set_ylabel('Y-COORDINATE OF RAW POCA')
ax3.set_zlabel('Z-COORDINATE OF RAW POCA')
ax3.set_title('3D VISUALIZATION OF RAW POCA')

ax4=fig.add_subplot(2,2,4,projection='3d')
ax4.scatter(xaxis_filtered,yaxis_filtered,zaxis_filtered,zdir='z',s=2)
ax4.set_xlabel('X-COORDINATE OF FILTERED POCA')
ax4.set_ylabel('Y-COORDINATE OF FILTERED POCA')
ax4.set_zlabel('Z-COORDINATE OF FILTERED POCA')
ax4.set_title('3D VISUALIZATION OF FILTERED POCA')

plt.show()

