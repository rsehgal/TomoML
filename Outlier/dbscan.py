'''This program is about the implementation of the DBSCAN i.e. Density-based Spatial Clustering of Applications with Noise on the raw dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''
'''import sys
print(sys.path)
sys.path.append("/home/TomoML/Outlier/")'''
#Import sklearn for implementing DBSCAN and suitable data normalization
import numpy as np

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import MinMaxScaler

#Import numpy,pandas,visualization Modules from visualization_function program
from visualization_funcs import *

def dbscan_fit1(datafr):
	dbsc = DBSCAN(eps=0.27,metric='manhattan',min_samples=13000)
	print(datafr)
	dbsc.fit(datafr)
	labels = dbsc.labels_
	return labels
def dbscan_fit2(datafr):
	dbsc = DBSCAN(eps=0.31,metric='manhattan',min_samples=13000)
	print(datafr)
	dbsc.fit(datafr)
	labels = dbsc.labels_
	return labels

#Storing dataset in dataframe data structure of pandas
#raw_dataframe=pd.read_csv('../dataset/filtered.csv')
raw_dataframe=readfile('../dataset/raw.csv')
#Extracting individual series of attributes from the whole dataframe
xaxis=raw_dataframe['X']
yaxis=raw_dataframe['Y']
zaxis=raw_dataframe['Z']
scat_angel=raw_dataframe['scatteringAngle']


#Visualization

#fig=plt.figure()
'''ax=fig.add_subplot(2,2,1,projection='3d')
colorbar=ax.scatter(xaxis,yaxis,zaxis,c=scat_angel,cmap=plt.magma(),s=2)
fig.colorbar(colorbar)'''


#Normalization
raw_dataframe = raw_dataframe.as_matrix().astype("float32", copy = False)
raw_dataframe=raw_dataframe[:,:-1]
stscaler=MinMaxScaler(feature_range=(0,1))
raw_dataframe = stscaler.fit_transform(raw_dataframe)
col_names1=['X','Y','Z']
raw_fr=pd.DataFrame(raw_dataframe,columns=col_names1)
print(raw_fr.describe())
print(type(raw_dataframe))

print(raw_dataframe)
#dbsc = DBSCAN(eps=0.3,metric='manhattan',min_samples=8800)
#dbsc.fit(raw_dataframe)
#labels = dbsc.labels_

print(raw_dataframe[:,0:2])
print(raw_dataframe[:,1:])
print(raw_dataframe[:,[0,2]])

raw_xlabel='X COORDINATE OF RAW POCA'
raw_ylabel='Y COORDINATE OF RAW POCA'
raw_zlabel='Z COORDINATE OF RAW POCA'

'''twodimensional_plot(raw_fr['X'],raw_fr['Z'],raw_xlabel,raw_ylabel)
twodimensional_plot(raw_fr['Y'],raw_fr['Z'],raw_ylabel,raw_zlabel)
twodimensional_plot(raw_fr['X'],raw_fr['Z'],raw_xlabel,raw_zlabel)
display()'''
#labels1=dbscan_fit1(raw_dataframe[:,0:2])
labels2=dbscan_fit2(raw_dataframe[:,1:2])
#labels3=dbscan_fit(raw_dataframe[:,[0]]

print(len(labels2[labels2==0]))

'''fin_labels1=np.column_stack((raw_dataframe,labels1))
plot_labels1=fin_labels1[fin_labels1[:,3]==0]
print(plot_labels1)
twodimensional_plot(plot_labels1[:,0],plot_labels1[:,1],raw_xlabel,raw_ylabel)
display()'''
#print(labels2)
'''#Total no of clusters
print(labels)
print(len(labels))
final=np.column_stack((raw_dataframe,labels))
plot=final[final[:,3]==0]
print(plot)
col_names2=['X','Y','Z','Labels']
print(col_names2)
final_df=pd.DataFrame(plot,columns=col_names2)
print(final_df.describe())	
ax1=fig.add_subplot(1,1,1,projection='3d')
ax1.scatter(final_df['X'],final_df['Y'],final_df['Z'],zdir='z',s=2)

print("TOtal no of clusters=",len(set(labels)))
#print(labels[labels>=0].sum())
plt.show()
'''

