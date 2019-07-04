'''This is the k-nearest neighbours classification problem.
Lazy and efficient approach for classification'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from visualization_funcs import *

import pandas as pd
import numpy as np
import math
from visualization_funcs import *
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

training_set=pd.read_csv("../dataset/training.csv",header=None)
#print(training_set)
info_training=training_set.describe()
count_training_set=int(info_training.loc['count',0])
k=int(math.sqrt(count_training_set))
if(k%2==0):
    k=k+1
list_k=list(range(1,k))
neighbors=filter(lambda x: x%2!=0,list_k)
test_set=pd.read_csv("../dataset/testing.csv",header=None)


#EXtracting the features from test and training datasets
training_features=training_set.iloc[:,0:6]
#print("hi i am Aniket",training_features)
target_features=training_set.iloc[:,9]
#print(target_features)
test_features=test_set.iloc[:,0:6]
testing_target_features=test_set.iloc[:,9]

cv_scores=[]

#Applying kneigbors classification

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors = k,weights='uniform')
    scores = cross_val_score(knn, training_features, target_features, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

misclassification_error=[1-x for x in cv_scores]
ele=min(misclassification_error)
index_minimum=misclassification_error.index(ele)
#print(index_minimum)
'''twodimensional_plot(neighbors,misclassification_error)
display()
'''
knn = KNeighborsClassifier(n_neighbors = 29,weights='uniform')
#Fit The Model

knn.fit(training_features,target_features)

#Accuracy of the Model
prediction = knn.predict(test_features)

print(testing_target_features,"\n")
print(prediction)
print("PREDICTION OF KNN IS:",knn.score(test_features,testing_target_features))


#Plotting a ROC Curve
y_scores = knn.predict_proba(test_features)
fpr, tpr, threshold = roc_curve(testing_target_features, y_scores[:,1])
roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

