'''This program plots the histogram for Distance of Closest Approach between two tracks.''' 


#Import visualization functions
 
from visualization_funcs import *

from beautify_data import *
#Import pandas,numpy
import pandas as pd
import numpy as np


filtered_df=readfile('../dataset/filteredWithDoCA.csv')

print(filtered_df)
print(filtered_df.head())
print(filtered_df.info())
print(filtered_df.describe())


df_doca=filtered_df[['doca']]
mean_doca_df=np.mean(df_doca)

k=str(mean_doca_df)
k1=k[8:15]

stddev_doca_df=np.std(df_doca)
d=str(stddev_doca_df)
d1=d[8:15]

count_doca_df=len(df_doca)
w=str(count_doca_df)


#legend={'Entries':count_doca_df,'Mean':mean_doca_df,'Std. Deviation':stddev_doca_df}
attributes=("Entries","Mean","Std_Deviation")
text="Entries:\t"+w+"\n"+"Mean:\t"+k1+"\n"+"Std Deviation:\t"+d1
plot_hist(filtered_df['doca'],"Distance of Closest Approach","Frequency Count",text,"Distance of Closest Approach")

display()

