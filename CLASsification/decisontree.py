'''This is the Decision Tree Algorithm.
Assumptions about distribution are not caalculated.'''

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from visualization_funcs import *

import pandas as pd
import numpy as np
import math


training_set=pd.read_csv("../dataset/training.csv",header=None)
#print(training_set)
info_training=training_set.describe()
count_training_set=int(info_training.loc['count',0])
k=int(math.sqrt(count_training_set))

test_set=pd.read_csv("../dataset/testing.csv",header=None)


#EXtracting the features from test and training datasets
training_features=training_set.iloc[:,3:6]
#training_features=training_features.reshape(-1,1)
#print("hi i am Aniket",training_features)
target_features=training_set.iloc[:,9]
#print(target_features)
test_features=test_set.iloc[:,3:6]
#test_features=test_features.reshape(-1,1)
testing_target_features=test_set.iloc[:,9]



#Create a Gaussian Classifier
clf=DecisionTreeClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(training_features,target_features)

important_features=clf.feature_importances_

print(important_features)

predict_final_result=clf.predict(test_features)

print("Accuracy:",metrics.accuracy_score(testing_target_features,predict_final_result))


