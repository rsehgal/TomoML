


#import all the modules
import pandas as pd
import numpy as np
from pyod.models.abod import ABOD
from sklearn.preprocessing import MinMaxScaler


#Read the data and store it in a dataframe
poca_df=pd.read_csv('../dataset/filtered.csv')
print(poca_df)



#Check for missing Values
print(poca_df.info())


#print its 5 point summary
print(poca_df.describe())


#Normalize the values
scaler = MinMaxScaler(feature_range=(0, 1))
poca_df = scaler.fit_transform(poca_df)

col_names=['X','Y','Z','scatteringAngle']
poca_df=pd.DataFrame(poca_df)

poca_df.columns=col_names

print(poca_df)





#Store values in numpy arrays for later use
X = poca_df['X'].values.reshape(-1,1)
Y=poca_df['Y'].values.reshape(-1,1)
Z =poca_df['Z'].values.reshape(-1,1)
scatteringAngle=poca_df['scatteringAngle'].values.reshape(-1,1)
fin = np.concatenate((X,Y),axis=1)
#print(fin)




#Apply the ABOD model for classifying outliers from dataset
classifier=ABOD(contamination=0.05)
classifier.fit(fin)

#Apply Raw anomaly score
score=classifier.decision_function(fin) * -1

#Prediction of a datapoint whether it is an outlier or inlier
poca_predict=classifier.predict(fin)

print(poca_predict)







