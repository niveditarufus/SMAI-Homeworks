import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
fig, ax = plt.subplots(2, 2)
 
x = np.array([[-2,4],[4,1],[1,6],[2,5]])
y = np.array([0,0,1,1])

xtest = np.array([[2,0],[0,1],[3,-0.5],[4,4],[-3,1]])
ytest = np.array([1,1,1,1,-1])

sc = StandardScaler()
sc.fit(x)
x = sc.transform(x)
sc.fit(xtest)
# xtest = sc.transform(xtest)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(x, y)
y_pred = ppn.predict(xtest)
print('Accuracy Perceptron: %.2f' % accuracy_score(ytest, y_pred))
w = ppn.coef_[0]
bias = ppn.intercept_[0]
a = np.arange(-5,5,1)
b = -w[0]*a - bias
b = b/w[1]
ax[0,0].set_title('Perceptron')
ax[0,0].scatter(x[:,0], x[:,1], c=y, cmap='Paired_r', edgecolors='k');
ax[0,0].scatter(xtest[:,0],xtest[:,1], color = 'Green', marker = '^',label = 'test data')
ax[0,0].legend()
ax[0,0].plot(a,b)


log_reg = LogisticRegression(solver = 'lbfgs')
log_reg.fit(x,y)
y_pred = log_reg.predict(xtest)
print('Accuracy Logistic Regression gamma = 1: %.2f' % accuracy_score(ytest, y_pred))
w = log_reg.coef_[0]
bias = log_reg.intercept_[0]
a = np.arange(-4,4,1)
b = -w[0]*a - bias
b = b/w[1]
ax[0,1].set_title('LogisticRegression gamma = 1')
ax[0,1].scatter(x[:,0], x[:,1], c=y, cmap='Paired_r', edgecolors='k');
ax[0,1].scatter(xtest[:,0],xtest[:,1], color = 'Green', marker = '^',label = 'test data')
ax[0,1].legend()
ax[0,1].plot(a,b)

gamma = 0.001
x1 = gamma*x
sc = StandardScaler()
sc.fit(x1)
log_reg = LogisticRegression(solver = 'lbfgs')
log_reg.fit(x1,y)
y_pred = log_reg.predict(xtest)
print('Accuracy Logistic Regression gamma = 0.001: %.2f' % accuracy_score(ytest, y_pred))
w = log_reg.coef_[0]
bias = log_reg.intercept_[0]
a = np.arange(-4,4,1)
b = -w[0]*a - bias
b = b/w[1]
x1 = sc.transform(x1)
ax[1,0].set_title('Logistic Regression gamma = 0.001')
ax[1,0].scatter(x[:,0], x[:,1], c=y, cmap='Paired_r', edgecolors='k');
ax[1,0].scatter(xtest[:,0],xtest[:,1], color = 'Green', marker = '^',label = 'test data')
ax[1,0].legend()
ax[1,0].plot(a,b)

gamma = 1000
x1 = gamma*x
sc = StandardScaler()
sc.fit(x1)
log_reg = LogisticRegression(solver = 'lbfgs')
log_reg.fit(x1,y)
y_pred = log_reg.predict(xtest)
print('Accuracy Logistic Regression gamma = 1000: %.2f' % accuracy_score(ytest, y_pred))
w = log_reg.coef_[0]
bias = log_reg.intercept_[0]
a = np.arange(-4,4,1)
b = -w[0]*a - bias
b = b/w[1]
x1 = sc.transform(x1)
ax[1,1].set_title('Logistic Regression gamma = 1000')
ax[1,1].scatter(x[:,0], x[:,1], c=y, cmap='Paired_r', edgecolors='k');
ax[1,1].scatter(xtest[:,0],xtest[:,1], color = 'Green', marker = '^',label = 'test data')
ax[1,1].legend()
ax[1,1].plot(a,b)

plt.show()