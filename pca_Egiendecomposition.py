# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import plotly
plotly.tools.set_credentials_file(username='ShirleyLi', api_key='7OZepfhXJgstSoL2vh5p')

# Importing the dataset
dataset = pd.read_csv('/Users/Shirley/Desktop/Gender_Recognition_by_Voice/voice.csv')
X= dataset.iloc[:,0:-1].values
y= dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X_train)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# get the covariance matrix
mean_vector = np.mean(X_std, axis=0)
covariance_matrix = (X_std - mean_vector).T.dot((X_std- mean_vector))/(X_std.shape[0]-1)
print('Covariance matrix \n%s' %covariance_matrix)

cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

cor_mat1 = np.corrcoef(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

u,s,v = np.linalg.svd(X_std.T)
u

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
trace1 = Bar(
        x=['PC %s' %i for i in range(1,20)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,20)],
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.iplot(fig)
