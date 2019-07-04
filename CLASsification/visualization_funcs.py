'''All the dependent functions such as reading a .csv file,visualizing two dimensional plots,visualizing three dimensional plots are covered int this code'''
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
plt.title("2D AND 3D VISUALIZATION OF POCA")

#Function for making 2D plots 
'''Two axes will be used for plotting the graph so the parameters are axis1,axis2.Provision for providing labels to the respective axes is also given.Here we are assuming that 4 graphs need to be plotted so we are dividing plot space into 4 subdivisions i.e. 2 rows and 2 columns'''
def twodimensional_plot(axis1,axis2,xlabel='AXIS1',ylabel='AXIS2'):
	if(twodimensional_plot.counter>4):
		print("Max plotting limit reached.....")
	else:
		ax1=fig.add_subplot(2,2,twodimensional_plot.counter)
		ax1.scatter(axis1,axis2,s=2)
		ax1.set_xlabel(xlabel)
		ax1.set_ylabel(ylabel)
		twodimensional_plot.counter+=1
twodimensional_plot.counter=1





def cluster_plot(color,axis1,axis2,xlabel='AXIS1',ylabel='AXIS2'):
	if(cluster_plot.counter>4):
		print("Max plotting limit reached.....")
	else:
		ax1=fig.add_subplot(2,2,cluster_plot.counter)
		ax1.scatter(axis1,axis2,s=2,c=color)
		ax1.set_xlabel(xlabel)
		ax1.set_ylabel(ylabel)
		cluster_plot.counter+=1
cluster_plot.counter=1

#Function for making 3D plots 
'''Three axes will be used for plotting the graph so the parameters are axis1,axis2,axis3.Provision for providing labels to the respective axes is also given.Here we are assuming that 4 graphs need to be plotted so we are dividing plot space into 4 subdivisions i.e. 2 rows and 2 columns'''
def threedimensional_plot(color,axis1,axis2,axis3,xlabel='AXIS1',ylabel='AXIS2',zlabel='AXIS3'):
	if(twodimensional_plot.counter>4):
		print("Max plotting limit reached.....")
	else:
		ax1=fig.add_subplot(2,2,threedimensional_plot.counter,projection='3d')
		ax1.scatter(axis1,axis2,axis3,s=2,c=color)
		ax1.set_xlabel(xlabel)
		ax1.set_ylabel(ylabel)
		ax1.set_zlabel(zlabel)

		threedimensional_plot.counter+=1
threedimensional_plot.counter=3

def plot_hist(x_d_Axis,xlabel,ylabel,text_to_be_added,title):
	ax3=fig.add_subplot(1,1,1)
	ax3.hist(x_d_Axis)
	ax3.set_xlabel(xlabel)
	ax3.set_ylabel(ylabel)
	ax3.legend(text_to_be_added,ncol=2,title=title,fancybox=True,facecolor='yellow')
	ax3.text(40,12000,text_to_be_added,fontsize=8,bbox=dict(facecolor='yellow', alpha=0.5))
	
#Display the results
def display():
	plt.show()


