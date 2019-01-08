# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 20:34:58 2019

@author: Animesh
"""
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
y=[]
from feature_selector import FeatureSelector
# Features are in train and labels are in train_labels
fs = FeatureSelector(data = train, labels = train_labels)