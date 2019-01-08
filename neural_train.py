# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:36:09 2019

@author: Animesh
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import joblib    
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

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

#adding layers
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'tanh', input_dim=22))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'tanh'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'tanh'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])


# Fitting our model 
classifier.fit(x, y, batch_size = 10, epochs = 100)
joblib.dump(classifier, 'classifier/neural_network.pkl',compress=9)