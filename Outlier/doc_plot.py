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


stddev_doca_df=np.std(df_doca)


count_doca_df=len(df_doca)


#legend={'Entries':count_doca_df,'Mean':mean_doca_df,'Std. Deviation':stddev_doca_df}

legend=[count_doca_df,mean_doca_df,stddev_doca_df]
plot_hist(filtered_df['doca'],"Distance of Closest Approach","Frequency Count",legend,"Distance of Closest Approach")

display()

