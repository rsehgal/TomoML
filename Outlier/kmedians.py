from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES

#Import pandas and numpy for data accessing and manipulation
import numpy as np
import pandas as pd

#IMPOrt visualization functions
from visualization_funcs import *

# Load list of points for cluster analysis.
sample = read_sample("../dataset/filteredDiffMaterial.txt")

#Storing dataset in dataframe data structure of pandas
poca_dataframe=readfile('../dataset/CSVfilteredDiffMaterial.csv')
corr=poca_dataframe.corr(method='pearson')


#COnversion toa NUMPY ARRaY
poca_dataframe = poca_dataframe.as_matrix().astype("float32", copy = False)
#new_poca_dataframe=poca_dataframe[:,0:3]
# Create instance of K-Medians algorithm.
#initial_medians = [[0.0, 0.1], [2.5, 0.7]]
initial_medians_indices=np.random.randint(0,poca_dataframe.shape[0],4)
initial_medians=poca_dataframe[initial_medians_indices]
initial_medians=initial_medians.tolist()

kmedians_instance = kmedians(sample, initial_medians)

column_names=["X","Y","Z","Scat_Angle","DoCA"]
column_names_final=["X","Y","Z","Scat_Angle","DoCA","mean_Scat_angle"]
filtered_poca_dataframe=pd.DataFrame(columns=column_names_final)
# Run cluster analysis and obtain results.
kmedians_instance.process()
clusters = kmedians_instance.get_clusters()
medians = kmedians_instance.get_medians()

mean_scat_angle_list=[]
for i in clusters:
	final_poca_dataframe=poca_dataframe[i]
	
	final_poca_dataframe=pd.DataFrame(final_poca_dataframe,columns=column_names)
	
	info_final_poca_dataframe=final_poca_dataframe.describe()	
	scat_angle=info_final_poca_dataframe.loc['mean']['Scat_Angle']
	scat_angle="{0:.2f}".format(scat_angle)
	#final_poca_dataframe['mean_Scat_angle']=info_final_poca_dataframe.loc['mean']['Scat_Angle']
	final_poca_dataframe['mean_Scat_angle']=scat_angle
	print(final_poca_dataframe)
	#mean_scat_angle_list.append(info_final_poca_dataframe.loc['mean']['Scat_Angle'])
	mean_scat_angle_list.append(scat_angle)
	filtered_poca_dataframe=pd.concat([filtered_poca_dataframe,final_poca_dataframe])
		
print(filtered_poca_dataframe)
xaxis=filtered_poca_dataframe["X"]
yaxis=filtered_poca_dataframe["Y"]
zaxis=filtered_poca_dataframe['Z']
mean_scat_angle=filtered_poca_dataframe['mean_Scat_angle']

cluster1_df=filtered_poca_dataframe[filtered_poca_dataframe['mean_Scat_angle']==mean_scat_angle_list[0]]
cluster2_df=filtered_poca_dataframe[filtered_poca_dataframe['mean_Scat_angle']==mean_scat_angle_list[1]]
cluster3_df=filtered_poca_dataframe[filtered_poca_dataframe['mean_Scat_angle']==mean_scat_angle_list[2]]
cluster4_df=filtered_poca_dataframe[filtered_poca_dataframe['mean_Scat_angle']==mean_scat_angle_list[3]]

info_cluster1_df=cluster1_df.describe()
info_cluster2_df=cluster2_df.describe()
info_cluster3_df=cluster3_df.describe()
info_cluster4_df=cluster4_df.describe()

print(info_cluster1_df)

'''

info_cluster1_df.to_csv("../dataset/kmedians_cluster1.csv")
info_cluster2_df.to_csv("../dataset/kmedians_cluster2.csv")
info_cluster3_df.to_csv("../dataset/kmedians_cluster3.csv")
info_cluster4_df.to_csv("../dataset/kmedians_cluster4.csv")
'''

threedimensional_plot(mean_scat_angle,xaxis,yaxis,zaxis)
#cluster_plot(mean_scat_angle,yaxis,zaxis)
display()
'''
# Visualize clustering results.
visualizer = cluster_visualizer_multidim()
visualizer.append_clusters(clusters, sample)
visualizer.append_cluster(initial_medians, marker='*', markersize=10)
visualizer.append_cluster(medians, marker='*', markersize=10)
visualizer.show(max_row_size=3)
'''


