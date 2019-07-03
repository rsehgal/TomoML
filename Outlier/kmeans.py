'''This program is about the implementation of the K-Means Clustering Algorithm on the poca dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''
import sys
from math import *
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
poca_dataframe=readfile("../dataset/CSVfilteredDiffMaterial.csv")
corr=poca_dataframe.corr(method='pearson')

#Extracting individual series of attributes from the whole dataframe
xaxis=poca_dataframe['X']
yaxis=poca_dataframe['Y']
zaxis=poca_dataframe['Z']
scat_angle=poca_dataframe['Scat_Angle']
doca=poca_dataframe['DoCA']
print(poca_dataframe.describe())

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
sorted_docalabel="Sorted Doca"
docalabel="Distance Of Closest Approach(DocA)"
SIZE_label="SIZE of Cluster"
log_of_doca_label="log(doca)"
distance_label="CEntroid distance"
log_of_CEntroid_DIstance="log(centroid_distancE)"
log_of_sorted_doca_label="log(doca)(sorted)"

final_df=pd.DataFrame(final,columns=col_names2)
print(final_df.describe())	
'''twodimensional_plot(labels,xaxis,yaxis,xlabel,ylabel)
twodimensional_plot(labels,yaxis,zaxis,ylabel,zlabel)
twodimensional_plot(labels,xaxis,zaxis,xlabel,zlabel)'''


#Finding 5point summary of individual clusters
cluster1_df=final_df[final_df['Labels']==0.0]
cluster2_df=final_df[final_df['Labels']==1.0]
cluster3_df=final_df[final_df['Labels']==2.0]
cluster4_df=final_df[final_df['Labels']==3.0]
info_cluster_1=cluster1_df.describe()
scat_ANgle_1=info_cluster_1.loc['mean']['Scat_Angle']
scat_Angle_1="{0:.4f}".format(scat_ANgle_1)
info_cluster_1.to_csv("../dataset/kmeans_filtereddiffmaterial_cluster_1.csv")

info_cluster_2=cluster2_df.describe()
scat_ANgle_2=info_cluster_2.loc['mean']['Scat_Angle']
scat_Angle_2="{0:.4f}".format(scat_ANgle_2)
info_cluster_2.to_csv("../dataset/kmeans_filtereddiffmaterial_cluster_2.csv")

info_cluster_3=cluster3_df.describe()
scat_ANgle_3=info_cluster_3.loc['mean']['Scat_Angle']
scat_Angle_3="{0:.4f}".format(scat_ANgle_3)
info_cluster_3.to_csv("../dataset/kmeans_filtereddiffmaterial_cluster_3.csv")

info_cluster_4=cluster4_df.describe()
scat_ANgle_4=info_cluster_4.loc['mean']['Scat_Angle']
scat_Angle_4="{0:.4f}".format(scat_ANgle_4)
info_cluster_4.to_csv("../dataset/kmeans_filtereddiffmaterial_cluster_4.csv")

label_Dict={0.0:scat_Angle_1,1.0:scat_Angle_2,2.0:scat_Angle_3,3.0:scat_Angle_4}
final_df['MEan_SCATTering_Angle']=final_df['Labels'].map(label_Dict)
mean_Scat_ANgle=final_df['MEan_SCATTering_Angle']
print(final_df)


threedimensional_plot(mean_Scat_ANgle,xaxis,yaxis,zaxis,xlabel,ylabel,zlabel)
'''
final_df[final_df['Labels']==1.0]['MEAN_Scattering_ANgle']=scat_Angle_2
final_df[final_df['Labels']==2.0]['MEAN_Scattering_ANgle']=scat_Angle_3
final_df[final_df['Labels']==3.0]['MEAN_Scattering_ANgle']=scat_Angle_4


cluster4_df['Logarithm_of_Doca']=np.log(cluster4_df['doca'])
print(cluster4_df)

cluster4_df_1=cluster4_df.sort_values("doca")
sorted_Doca=cluster4_df_1['doca']

doca_cluster_4=cluster4_df['doca']
len_cluster=len(labels[labels==3.0])



minimum_X1=info_cluster_4.loc['min']['X']
maximum_X1=info_cluster_4.loc['max']['X']

centroid_x1=(minimum_X1+maximum_X1)/2

minimum_Y1=info_cluster_4.loc['min']['Y']
maximum_Y1=info_cluster_4.loc['max']['Y']


centroid_Y1=(minimum_Y1+maximum_Y1)/2

minimum_Z1=info_cluster_4.loc['min']['Z']
maximum_Z1=info_cluster_4.loc['max']['Z']

centroid_Z1=(minimum_Z1+maximum_Z1)/2

centroid=[centroid_x1,centroid_Y1,centroid_Z1]

cluster4_df['distance_from_centroid']=cluster4_df[['X', 'Y','Z']].sub(np.array(centroid)).pow(2).sum(1).pow(0.5)

print(cluster4_df)

cluster4_df_distance=cluster4_df.sort_values("distance_from_centroid")
sorted_cent_dist=cluster4_df_distance["distance_from_centroid"]


cluster_range=np.arange(0,len_cluster)
log_doca_cluster4=cluster4_df['Logarithm_of_Doca']

log_doca_cluster4_1=cluster4_df_1['Logarithm_of_Doca']


mean_scat_angle=info_cluster_4.loc['mean']['Scat_Angle']
std_scat_angle=info_cluster_4.loc['std']['Scat_Angle']
limit1=mean_scat_angle-2*std_scat_angle
limit2=mean_scat_angle+2*std_scat_angle
filtered_cluster4=cluster4_df[cluster4_df['Scat_Angle']>=limit1]
filtered_cluster4=filtered_cluster4[filtered_cluster4['Scat_Angle']<=limit2]
print(filtered_cluster4.describe())

#cluster_4_doca=cluster4_df['doca']
twodimensional_plot(doca_cluster_4,log_doca_cluster4,docalabel,log_of_doca_label)
twodimensional_plot(cluster_range,log_doca_cluster4,SIZE_label,log_of_doca_label)
twodimensional_plot(cluster_range,log_doca_cluster4_1,sorted_docalabel,log_of_sorted_doca_label)
#twodimensional_plot(cluster_range,sorted_cent_dist,distance_label,log_of_CEntroid_DIstance)
twodimensional_plot(filtered_cluster4['X'],filtered_cluster4['Y'])

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
