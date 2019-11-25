import numpy as np
import random
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)

def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1])], \
                     [cov(X[1], X[0]), cov(X[1], X[1])]])

m, b = 0.2, 0
lower, upper = -25, 25

x = np.array([1,2,1,2])
y = np.array([1,2,2,1])

print len(y),len(x)
X = np.vstack((x, y)).T

X = X - np.mean(X, 0)

b = np.linspace(-5,5,100)
a = b
#

print(cov_mat(X.T))

Y = X
# 
print(np.mean(Y,0))
cov_mat(Y.T)
C = cov_mat(Y.T)
eVe, eVa = np.linalg.eig(C)
print(eVe,eVa)
i=0
plt.scatter(Y[:, 0], Y[:, 1], c= 'b')
for e, v in zip(eVe, eVa.T):
    if i==0:
    	plt.plot([0, 3*np.sqrt(e)*v[0]], [0, 3*np.sqrt(e)*v[1]], 'r', lw=3)
    else:
    	plt.plot([0, 3*np.sqrt(e)*v[0]], [0, 3*np.sqrt(e)*v[1]], 'g', lw=3)
    i = i+1
# plt.plot(a, b, '-k')

plt.title('eigen vectors')
plt.axis('equal');
plt.show()

