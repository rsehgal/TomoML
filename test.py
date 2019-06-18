#Import pandas package for creating Dataframes
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#creating dataframe from the .csv file
raw_dataframe=pd.read_csv('raw.csv')
#scat_ang=raw_dataframe['X']

print(raw_dataframe.describe())

