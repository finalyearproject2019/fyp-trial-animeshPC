# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 20:11:02 2019

@author: Animesh
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.svm import SVC
    
results=[]
features=[]
file=open('Training Dataset.arff').read()
list=file.split('\n')
data=np.array(list)
data1=[i.split(',') for i in data]
data1=data1[0:-1]
for i in data1:
	results.append(i[30])
data1=np.array(data1)
features=data1[:,:-1]

x=features[:,[0,1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,22,23,24,25,27,29]]
y=results

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state =0)

#applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1, 100, 1000], 'gamma': [ 0.1, 0.2,0.3], 'random_state': [0,1,42]}]
grid_search = GridSearchCV(SVC(kernel='rbf' ),  parameters,cv =5)
grid_search.fit(x_train, y_train)

#printing best parameters 
#print("Best Accurancy =" +str( grid_search.best_score_))
print("best parameters =" + str(grid_search.best_params_)) 

#fitting kernal with the parameters obtained
classifier = SVC(C=1000, kernel = 'rbf', gamma = 0.2 , random_state = 0)
#classifier = SVC(C=10, kernel = 'rbf', gamma = 0.2 , random_state = 0)
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
    
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
mean_accuracy =  100*round(accuracies.mean(),4)
standard_deviation = round(accuracies.std(),4)
print("mean accuracy = "+ str(mean_accuracy))
print("standard deviation = "+ str(standard_deviation))


