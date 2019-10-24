import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator,  X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, '-', color="r",
             label="Training score")

    plt.legend()
    return plt


def sigmoid(z):
	return 1/(1+ np.exp(-z))

def sigmoid_derivative(z):
	return z*(1-z)	

class NeuralNetwork:
    def __init__(self, x, y,alpha=1,w = 'random'):
        self.input = x
        self.y = y.reshape(x.shape[0],1)
        self.output = np.zeros(self.y.shape)
        self.alpha = alpha
        self.input = x
        if w == 'zeros':
            self.weights1 = np.zeros((self.input.shape[1],2))
            self.weights2 = np.zeros((2,1))
        elif w == 'ones':
            self.weights1 = np.ones((self.input.shape[1],2))
            self.weights2 = np.ones((2,1))
        elif w == 'random':
           self.weights1 = np.random.rand(self.input.shape[1],2) 
           self.weights2 = np.random.rand(2,1)                 

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
       	self.output = sigmoid(self.y*np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, ((self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += self.alpha*d_weights1
        self.weights2 += self.alpha*d_weights2

def testing(X_test,y_test,weights1,weights2):
	t=0
	test = sigmoid(np.dot(X_test , weights1))
	test = sigmoid(np.dot(test, weights2))
	t = np.where(test>0.5,1,-1)
	t = np.equal(t,y_test.reshape(test.shape))
	t = t[np.where(t == True)]
	return test,len(t)


mean1 = [15, 25]
cov1 = [4, 1,], [1, 3]
mean2 = [10, 23]
cov2 = [2,1],[1,1]

x1 = np.random.multivariate_normal(mean1, cov1, 100).T
x2= np.random.multivariate_normal(mean2, cov2, 100).T

# fig= plt.figure()
# ax= fig.add_subplot(111)
# ax.scatter(x1[0],x1[1], c='r',marker ='o')
# ax.scatter(x1[0],x2[1], c='b',marker ='^')
# ax.set_xlabel("x - axis")
# ax.set_ylabel("y - axis")
# ax.set_title("Data")
# plt.show()

x2 = x2.T
x1 = x1.T
X1 = np.ones([100,3])
X2 = -1*np.ones([100,3])
X1[:,0:2] = x1
X2[:,0:2] = x2
X = np.vstack((X1,X2))

X_train, X_test, y_train, y_test = train_test_split(X[:,0:2], X[:,2], test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
random_wts = NeuralNetwork(X_train,y_train,0.1,'random')
zeros_wts = NeuralNetwork(X_train,y_train,0.1,'zeros')
one_wts = NeuralNetwork(X_train,y_train,0.1,'ones')
t1 = []
t2 = []
t3 = []
for i in range(1000):
    random_wts.feedforward()
    random_wts.backprop()
    zeros_wts.feedforward()
    zeros_wts.backprop()
    one_wts.feedforward()
    one_wts.backprop()
    
    test, count1 = testing(X_train, y_train, random_wts.weights1, random_wts.weights2)
    t1.append(count1)
    test, count2 = testing(X_train, y_train, zeros_wts.weights1, zeros_wts.weights2)
    t2.append(count2)
    test, count3 = testing(X_train, y_train, random_wts.weights1, one_wts.weights2)
    t3.append(count3)
    

test, count1 = testing(X_test, y_test, random_wts.weights1, random_wts.weights2)
test, count2 = testing(X_test, y_test, zeros_wts.weights1, zeros_wts.weights2)
test, count3 = testing(X_test, y_test, one_wts.weights1, zeros_wts.weights2)

print("Accuracy with random initialisation of weights = ", count1/40)
print("Accuracy with weights initialised as 0 = ", count2/40)
print("Accuracy with weights initialised as 1 = ", count3/40)


#plotting 
plt.gca()
plt.plot(np.asarray(t1)/160)
plt.title("training curve of the given neural network random initialisation of weights")
plt.show()

plt.gca()
plt.title("training curve of the given neural with weights initialised as 0")
plt.plot(np.asarray(t2)/160)
plt.show()

plt.gca()
plt.title("training curve of the given neural with weights initialised as 1")
plt.plot((np.asarray(t3))/160)
plt.show()