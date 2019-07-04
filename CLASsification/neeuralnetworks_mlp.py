'''This is the Neural Networks classification problem.
Direct Mapping is not done between input and output.MLP technique will be used for backpropogation'''

from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
from sklearn.preprocessing import label_binarize,StandardScaler
 
from sklearn.multiclass import OneVsRestClassifier

from visualization_funcs import *

import pandas as pd
import numpy as np
import math


training_set=pd.read_csv("../dataset/training.csv",header=None)
#print(training_set)
test_set=pd.read_csv("../dataset/testing.csv",header=None)

training_input_layer=training_set.iloc[:,0:6]

test_input_layer=test_set.iloc[:,0:6]

#Normalization of data
scaler = StandardScaler()  
scaler.fit(training_input_layer)
training_input_layer=scaler.transform(training_input_layer)
print(training_input_layer)

target_features=training_set.iloc[:,9]
testing_target_features=test_set.iloc[:,9]

scaler.fit(test_input_layer)
test_input_layer=scaler.transform(test_input_layer)
print(test_input_layer)

#Applying NEuRAl NEtworks CLassification

clf = MLPClassifier()
clf.fit(training_input_layer,target_features)

predict_final=clf.predict(test_input_layer)

print("Accuracy:",metrics.accuracy_score(testing_target_features,predict_final))
