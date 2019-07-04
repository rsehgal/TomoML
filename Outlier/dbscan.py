'''This program is about the implementation of the DBSCAN i.e. Density-based Spatial Clustering of Applications with Noise on the poca dataset of coordinates and deviation of POCA i.e. Points of Closest Approach'''
'''import sys
print(sys.path)
sys.path.append("/home/TomoML/Outlier/")'''
#Import sklearn for implementing DBSCAN and suitable data normalization
import numpy as np

from sklearn.cluster import DBSCAN

from sklearn import preprocessing

#Import numpy,pandas,visualization Modules from visualization_function program
from visualization_funcs import *

def dbscan_fit1(datafr):
	dbsc = DBSCAN(eps=0.27,metric='manhattan',min_samples=13000)
	print(datafr)
	dbsc.fit(datafr)
	labels = dbsc.labels_
	return labels
def dbscan_fit2(datafr):
	dbsc = DBSCAN(eps=0.25,metric='manhattan',min_samples=700)
	print(datafr)
	dbsc.fit(datafr)
	labels = dbsc.labels_
	return labels

#Storing dataset in dataframe data structure of pandas

poca_dataframe=readfile('../dataset/filteredWithDoCA.csv')
poca=preprocessing.scale(poca_dataframe)
col_names1=['X','Y','Z','Scat_Angle','doca']
poca_dataframe=pd.DataFrame(poca,columns=col_names1)
corr=poca_dataframe.corr(method='pearson')
print(poca_dataframe.describe())
print(corr)
#Extracting individual series of attributes from the whole dataframe
'''xaxis=poca_dataframe['X']
yaxis=poca_dataframe['Y']
zaxis=poca_dataframe['Z']
scat_angel=poca_dataframe['Scat_Angle']
doca=poca_dataframe['doca']'''



'''
#Visualization

#fig=plt.figure()
ax=fig.add_subplot(2,2,1,projection='3d')
colorbar=ax.scatter(xaxis,yaxis,zaxis,c=scat_angel,cmap=plt.magma(),s=2)
fig.colorbar(colorbar)
'''

'''#Normalization
poca_dataframe = poca_dataframe.as_matrix().astype("float32", copy = False)
poca_dataframe=poca_dataframe[:,:]
stscaler=StandardScaler()
poca_dataframe = stscaler.fit_transform(poca_dataframe)

poca_fr=pd.DataFrame(poca_dataframe,columns=col_names1)
xaxis=poca_fr['X']
yaxis=poca_fr['Y']
zaxis=poca_fr['Z']
scat_angle=poca_fr['Scat_Angle']
doca=poca_fr['doca']

print(poca_fr.describe())
print(type(poca_dataframe))
twodimensional_plot(xaxis,yaxis)
twodimensional_plot(yaxis,zaxis)
twodimensional_plot(xaxis,zaxis)
twodimensional_plot(zaxis,doca)
print(poca_dataframe)
dbsc = DBSCAN(eps=0.3,metric='manhattan',min_samples=8800)
dbsc.fit(poca_dataframe)
#labels = dbsc.labels_'''
'''
print(poca_dataframe[:,0:2])
print(poca_dataframe[:,1:])
print(poca_dataframe[:,[0,2]])

poca_xlabel='X COORDINATE OF poca POCA'
poca_ylabel='Y COORDINATE OF poca POCA'
poca_zlabel='Z COORDINATE OF poca POCA'

twodimensional_plot(poca_fr['X'],poca_fr['Z'],poca_xlabel,poca_ylabel)
twodimensional_plot(poca_fr['Y'],poca_fr['Z'],poca_ylabel,poca_zlabel)
twodimensional_plot(poca_fr['X'],poca_fr['Z'],poca_xlabel,poca_zlabel)
display()
#labels1=dbscan_fit1(poca_dataframe[:,0:2])
labels2=dbscan_fit2(poca_dataframe[:,1:])
#labels3=dbscan_fit(poca_dataframe[:,[0]]

print(len(labels2[labels2==0]))

fin_labels2=np.column_stack((poca_dataframe,labels2))
plot_labels2=fin_labels2[fin_labels2[:,3]==0]
print(plot_labels2)
twodimensional_plot(plot_labels2[:,0],plot_labels2[:,1],poca_xlabel,poca_ylabel)
display()
#print(labels2)
#Total no of clusters
print(labels)
print(len(labels))
final=np.column_stack((poca_dataframe,labels))
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
#Display
display()

