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
#print(training_set[9])

training_set=training_set.sort_values(9,axis=0,ascending=True)

#print(training_set[9])
#print(training_set) 
info_training=training_set.describe()
count_training_set=int(info_training.loc['count',0])
k=int(math.sqrt(count_training_set))
if(k%2==0):
    k=k+1
list_k=list(range(1,k))
neighbors=filter(lambda x: x%2!=0,list_k)

test_set=pd.read_csv("../dataset/testing.csv",header=None)
test_Set=test_set.sort_values(9,axis=0,ascending=True)
#print(test_set[9])

#EXtracting the features from test and training datasets
training_features=training_set.iloc[:,0:6]
#print("hi i am Aniket",training_features)
target_features=training_set.iloc[:,9]
#print(target_features)
target_variable=label_binarize(target_features,classes=[1,2,3])
#print(target_variable)
n_classes=target_variable.shape[1]

#print(target_features)
test_features=test_set.iloc[:,0:6]
testing_target_features=test_set.iloc[:,9]
#print(testing_target_features)
testing_target_features=label_binarize(testing_target_features,classes=[2,3,4])
#print(testing_target_features)
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

knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 29,weights='uniform'))
#Fit The Model

knn.fit(training_features,target_variable)

#Accuracy of the Model
prediction = knn.predict(test_features)

print(testing_target_features,"\n")
print(prediction)
print("PREDICTION OF KNN IS:",knn.score(test_features,testing_target_features))


#Plotting a ROC Curve
y_scores=knn.predict_proba(test_features)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testing_target_features[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(testing_target_features.ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")


plt.show()

