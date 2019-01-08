# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:26:14 2019

@author: Animesh
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
    
results=[]
features=[]
file=open('Training Dataset.arff').read()
list=file.split('\n')
data=np.array(list)
data=[i.split(',') for i in data]
data=data[0:-1]
for i in data:
	results.append(i[30])
data=np.array(data)
features=data[:,:-1]

features=np.array(features).astype(np.int32)
results=np.array(results).astype(np.int32)
x=features[:,[0,1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,22,23,24,25,27,29]]
y=results

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state =0)

classifier = GaussianNB()
 
classifier.fit(x_train, y_train)

#predicting the tests set result
y_pred = classifier.predict(x_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)

#classification report
print(classification_report(y_test, y_pred))

#accuracy
accuracy = round(100 * accuracy_score(y_test,y_pred),4)
print("accuracy = " + str(accuracy)) #What percent of your predictions were correct?
    