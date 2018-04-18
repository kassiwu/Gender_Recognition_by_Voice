#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:41:35 2018

@author: shirley
"""

# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/shirley/Desktop/voice_Kaggle.csv')
X= dataset.iloc[:,0:-1]
y= dataset.iloc[:,-1]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# After scaling
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
classifier = RandomForestClassifier(n_estimators = 30, min_samples_split = 2, min_samples_leaf = 5, criterion = 'gini', random_state =0)
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test,prediction)
print(accuracy)


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
classifier = RandomForestClassifier(n_estimators = 30, min_samples_split = 2, min_samples_leaf = 5, criterion = 'entropy', random_state =0)
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test,prediction)
print(accuracy)
