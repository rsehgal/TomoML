'''This program plots the histogram for Distance of Closest Approach between two tracks.''' 


#Import visualization functions
 
from visualization_funcs import *


#Import pandas,numpy
import pandas as pd
import numpy as np

def beautify_data(filename):
	#Import the dataset
	df=pd.read_csv('../dataset/'+filename,header=None)
	#df.drop(0,axis=1)
	#print(df)
	#df.shift(3,axis=0)
	#Add the required Columns
	df.columns=['X','Y','Z','Scat_Angle','doca']
	
	#df=df[:]
	#print(df)

	#Store the beautified data in a new csv file
	df.to_csv("../dataset/"+filename,index=False,index_label=False)
	return df

