#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:44:11 2018

@author: shirley
"""

# support vector machine

# import the lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('/Users/shirley/Desktop/voice_Kaggle.csv')
print("Total number of samples: {}".format(dataset.shape[0]))
print("Total number of male: {}".format(dataset[dataset.label == 'male'].shape[0]))
print("Total number of female: {}".format(dataset[dataset.label == 'female'].shape[0]))
print("Correlation between each feature")
print(dataset.corr())

x= dataset.iloc[:,0:-1]
y= dataset.iloc[:,-1]

# splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#svm with default parameter without scaling
from sklearn.svm import SVC
from sklearn import metrics
sc=SVC()
sc.fit(x_train, y_train)
prediction = sc.predict(x_test)
print('Accuracy for default parameter without scaling')
print(metrics.accuracy_score(y_test,prediction))

#svm with linear kernel without scaling
from sklearn.svm import SVC
from sklearn import metrics
sc=SVC(kernel = 'linear')
sc.fit(x_train, y_train)
prediction = sc.predict(x_test)
print('Accuracy for svm with Linear kernel without scaling')
print(metrics.accuracy_score(y_test,prediction))

#svm with rbf kernel without scaling
from sklearn.svm import SVC
from sklearn import metrics
sc=SVC(kernel = 'rbf')
sc.fit(x_train, y_train)
prediction = sc.predict(x_test)
print('Accuracy for svm with RBF kernel without scaling')
print(metrics.accuracy_score(y_test,prediction))

# scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# fitting svm to the training set
from sklearn.svm import SVC
print('Linear Kernel after scaling')
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_train)
print('Accuracy in training samples for SVM: ',  accuracy_score(y_train,prediction))
prediction2 = classifier.predict(x_test)
print('Accuracy in testing samples for SVM: ', accuracy_score(y_test, prediction2))


print('RBF Kernel')
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print('Accuracy in testing samples for SVM: ', accuracy_score(y_test, prediction))
prediction2 = classifier.predict(x_test)
print('Accuracy in testing samples for SVM: ', accuracy_score(y_test, prediction2))

print('Polynomial Kernel')
classifier = SVC(kernel= 'poly' , random_state = 0)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print('Accuracy in testing samples for SVM: ', accuracy_score(y_test, prediction))
prediction2 = classifier.predict(x_test)
print('Accuracy in testing samples for SVM: ', accuracy_score(y_test, prediction2))

# virtualizing training set results
from matplotlib.colors import ListedColormap
x_set = x_train
y_set = y_train

# Linear kernel with different C values
from sklearn.model_selection import train_test_split, cross_val_score

def linear_c(x_train, y_train, c):
    accurancy = []
    for i in c:
        sc = SVC(kernel='linear', C=i)
        scores = cross_val_score(sc, x_train, y_train, cv=10, scoring='accuracy')
        accurancy.append(scores.mean())
        print("When the C value is: %d then the accurancy is: %f" % (i, scores.mean()))

    plt.plot(c, accurancy)
    plt.xlabel('Differnet C Values')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

c_values = list(np.arange(1, 16))

linear_c(x_train, y_train, c_values)

