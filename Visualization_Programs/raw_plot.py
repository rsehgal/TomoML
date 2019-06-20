#Import pandas package for creating Dataframes
import pandas as pd
#Import matplotlib for visualization
import matplotlib.pyplot as plt

#creating dataframe from the .csv file
raw_dataframe=pd.read_csv('raw.csv')


#Selecting series out of the Dataframe i.e. x-xoordinate of poca
xaxis=raw_dataframe['X']
print(xaxis)

#Selecting series out of the Dataframe i.e. y-coordinate of poca
yaxis=raw_dataframe['Y']

#Selecting series out of Dataframe i.e. z-coordinate of poca
zaxis=raw_dataframe['Z']

#Visualization
#colors=['red','green','blue']
plt.scatter(xaxis,yaxis,s=5)
plt.xlabel('X-COORDINATE OF POCA')
plt.ylabel('Y-COORDINATE OF POCA')
plt.show()
