'''This is the k-nearest neighbours classification problem.
Lazy and efficient approach for classification'''

from sklearn.neighbors import KNeighborsClassifier

from visualization_funcs import *

import pandas as pd
import numpy as np
import math

training_set=pd.read_csv("../dataset/training.csv",header=None)
#print(training_set)
info_training=training_set.describe()
count_training_set=info_training.loc['count',0]
k=int(math.sqrt(count_training_set))
test_set=pd.read_csv("../dataset/testing.csv",header=None)

#EXtracting the features from test and training datasets
training_features=training_set.iloc[:,0:6]
#print("hi i am Aniket",training_features)
target_features=training_set.iloc[:,9]
#print(target_features)
test_features=test_set.iloc[:,0:6]
testing_target_features=test_set.iloc[:,9]

#Applying kneigbors classification
knn = KNeighborsClassifier(n_neighbors = k,weights='uniform')

#Fit The Model
knn.fit(training_features,target_features)

#Accuracy of the Model
prediction = knn.predict(test_features)

print(testing_target_features,"\n")
print(prediction)
print("PREDICTION OF KNN IS:",knn.score(test_features,testing_target_features))
