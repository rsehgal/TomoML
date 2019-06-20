import numpy as np
from visualization_funcs import *
from sklearn.preprocessing import MinMaxScaler

df=readfile("../dataset/raw.csv")

raw_dataframe = df.as_matrix().astype("float32", copy = False)
raw_dataframe=raw_dataframe[:,:-1]
stscaler=MinMaxScaler(feature_range=(0,1))
raw_dataframe = stscaler.fit_transform(raw_dataframe)
col_names1=['X','Y','Z']
raw_fr=pd.DataFrame(raw_dataframe,columns=col_names1)
print(raw_fr.describe())
print(type(raw_dataframe))

data_x=raw_dataframe[:,0]
print(data_x)
data_y=raw_dataframe[:,1]
print(data_y)

data_xy=np.sqrt(np.sum((data_x-data_y)**2))

print(np.linalg.norm(data_x-data_y))
print(data_xy)
L1=[0.499451,0.497550]
raw_fr['distance']=raw_fr[['Y','Z']].sub(np.array(L1)).abs().sum(axis=1)
print(raw_fr)

#dist=raw_fr['distance']
dist=raw_fr.as_matrix().astype("float32",copy=False)
dist=dist[:,-1]
print(dist)
#mean
print(np.mean(dist))
