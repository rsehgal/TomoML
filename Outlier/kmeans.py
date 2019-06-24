'''This program is about the implementation of the K-Means Clustering Algorithm on the poca dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''
import sys

sys.path.append("/home/TomoML/'Visualization Programs'/")
#Import sklearn for implementing DBSCAN and suitable data normalization
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
from visualization_funcs import *

#Import pandas and numpy for data accessing and manipulation
import numpy as np
import pandas as pd


#Storing dataset in dataframe data structure of pandas
#poca_dataframe=pd.read_csv('../dataset/filtered.csv')
poca_dataframe=readfile('../dataset/filteredWithDoCA.csv')
corr=poca_dataframe.corr(method='pearson')

#Extracting individual series of attributes from the whole dataframe
xaxis=poca_dataframe['X']
yaxis=poca_dataframe['Y']
zaxis=poca_dataframe['Z']
scat_angle=poca_dataframe['Scat_Angle']
doca=poca_dataframe['doca']
print(poca_dataframe)

#COnversion toa NUMPY ARRaY
poca_dataframe = poca_dataframe.as_matrix().astype("float32", copy = False)

'''#Visualization

fig=plt.figure()
ax=fig.add_subplot(2,2,1,projection='3d')
colorbar=ax.scatter(xaxis,yaxis,zaxis,c=scat_angel,cmap=plt.magma(),s=2)
fig.colorbar(colorbar)



'''
print(poca_dataframe)

#Clustering
kmeans=KMeans(n_clusters=4,precompute_distances='auto')
kmeans.fit(poca_dataframe[:,:])
labels=kmeans.labels_


#Total no of clusters
print(labels)
print(len(labels))
final=np.column_stack((poca_dataframe,labels))
print(final)
#plot3=final[final[:,3]==2]
#print(plot1)
#print(plot2)
#print(plot3)
col_names2=['X','Y','Z','Scat_Angle','doca','Labels']
print(col_names2)
xlabel="X-AXIS OF FILTERED POCA"
ylabel="Y-AXIS OF FILTERED POCA"
zlabel="Z- AXIS OF FILTERED POCA"
final_df=pd.DataFrame(final,columns=col_names2)
print(final_df.describe())	
'''twodimensional_plot(labels,xaxis,yaxis,xlabel,ylabel)
twodimensional_plot(labels,yaxis,zaxis,ylabel,zlabel)
twodimensional_plot(labels,xaxis,zaxis,xlabel,zlabel)'''

threedimensional_plot(labels,xaxis,yaxis,zaxis,xlabel,ylabel,zlabel)

'''
ax1=fig.add_subplot(2,2,1,projection='3d')
ax1.scatter(final_df1['X'],final_df1['Y'],final_df1['Z'],zdir='z',s=2)

final_df2=pd.DataFrame(plot2,columns=col_names2)
print(final_df2.describe())	
ax2=fig.add_subplot(2,2,2,projection='3d')
ax2.scatter(final_df2['X'],final_df2['Y'],final_df2['Z'],zdir='z',s=2)

#final_df3=pd.DataFrame(plot3,columns=col_names2)
#print(final_df3.describe())	
#ax3=fig.add_subplot(2,2,3,projection='3d')
#ax3.scatter(final_df3['X'],final_df3['Y'],final_df3['Z'],zdir='z',s=2)

print("TOtal no of clusters=",len(set(labels)))
#print(labels[labels>=0].sum())
plt.show()
'''
#Display
display()
