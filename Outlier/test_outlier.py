import numpy as np
from visualization_funcs import *
from sklearn.preprocessing import MinMaxScaler

df=readfile("../dataset/raw.csv")

label1=np.array([1,0,0,0,1,0])
label2=np.array([1,1,0,1,0,1])
print(label1==label2.all())
print(np.array_equal(label1,label2))
