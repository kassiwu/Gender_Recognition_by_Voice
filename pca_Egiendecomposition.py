#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 23:21:18 2018

@author: shirley
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
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
X_std = sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# get the covariance matrix
mean_vector = np.mean(X_std, axis=0)
covariance_matrix = (X_std - mean_vector).T.dot((X_std- mean_vector))/(X_std.shape[0]-1)
print('Covariance matrix \n%s' %covariance_matrix)

# perform eigendecomposition basd on the covariance matrix of scaled data
covariance_matrix1 = np.corrcoef(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix1)
print('Eigen vlaues based on scaled data \n%s' %eig_vals)
print('Eigen vectors based on scaled data \n%s' %eig_vecs)

# perform eigendecomposition basd on the covariance matrix of raw data
covariance_matrix2 = np.corrcoef(X.T)
eig_vals2, eig_vecs2 = np.linalg.eig(covariance_matrix2)
print('Eigen vlaues based on raw data \n%s' %eig_vals2)
print('Eigen vectors based on raw data \n%s' %eig_vecs2)