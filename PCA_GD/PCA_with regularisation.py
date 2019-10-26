import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
    return (data, labels)

def normal_pca(X,k=2):
    X -= np.mean(X,axis = 0)
    C = np.cov(X.T)
    eig_vals, eigen_vec = np.linalg.eig(C)
    eigen_vec = eigen_vec[:,:k]
    Xnew = eigen_vec.T.dot(X.T)
    Xnew[0,:] *= -1 
    Xnew = Xnew.T
    return Xnew,eigen_vec

def gradient_descent_pca(X,initial_guess,k=2,n=1):
    C = np.cov(X.T)
    v = initial_guess
    for i in range(k):
        for j in range(100):
            v[:,i] = v[:,i] + n*(X.T.dot(X.dot(v[:,i])))
            v[:,i] /= np.linalg.norm(v[:,i])
    Xnew = v.T.dot(X.T)
    Xnew[0,:] *= -1 
    Xnew = Xnew.T
    return Xnew




train_data, train_labels = read_data("mnist_train.csv")
x0, a = normal_pca(train_data)
lr_model = LinearRegression()
lr_model.fit(x0, train_labels)
print('Reprojection Error: RIDGE Normal PCA : {}'.format(lr_model.score(x0, train_labels)))
# initial_guess = 0.4*np.ones([784,2])/np.sqrt(784)
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.3, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(x0, train_labels)
print('Reprojection Error: LASSO Normal PCA : {}'.format(lasso_pipe.score(x0, train_labels)))
initial_guess = a
x1 = gradient_descent_pca(train_data,initial_guess)
lr_model.fit(x1, train_labels)
print('Reprojection Error: RIDGE gradient descent PCA : {}'.format(lr_model.score(x1, train_labels)))
lasso_pipe.fit(x1, train_labels)
print('Reprojection Error: LASSO Normal PCA : {}'.format(lasso_pipe.score(x1, train_labels)))
