#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:04:19 2018

@author: shirley
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('/Users/shirley/Desktop/voice_Kaggle.csv')
X= dataset.iloc[:,0:-1]
y= dataset.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA


#Dimensionality reduction

print("Correlation between each feature")
print(dataset.corr())
sns.heatmap(dataset.corr())


n_components = 20

pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
