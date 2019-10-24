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

def tanh(z):
	return((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))

def sigmoid(z):
	return 1/(1+ np.exp(-z))

def sigmoid_derivative(z):
	return z*(1-z)

def tanh_derivative(z):
	return (1-(z**2))

class NeuralNetwork:
    def __init__(self, x, y,alpha=1,bias = False):
        self.bias = bias
        if bias == False:
        	self.input = x
        	self.weights1 = np.random.rand(self.input.shape[1],2) 
        	self.weights2 = np.random.rand(2,1)                 
        	self.y = y.reshape(x.shape[0],1)
        	self.output = np.zeros(self.y.shape)
        	self.alpha = alpha
        else:
        	temp = np.ones([x.shape[0],x.shape[1]+1])
        	temp[:,0:2] = x
        	self.input = x
        	self.weights1 = np.random.rand(self.input.shape[1],3) 
        	self.weights2 = np.random.rand(3,1)                 
        	self.y = y.reshape(x.shape[0],1)
        	self.output = np.zeros(self.y.shape)
        	self.alpha = alpha

    def feedforward(self):
        self.layer1 = tanh(np.dot(self.input, self.weights1))
       	self.output = sigmoid(self.y*np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, ((self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * tanh_derivative(self.layer1)))
        self.weights1 += self.alpha*d_weights1
        self.weights2 += self.alpha*d_weights2

def testing(X_test,y_test,weights1,weights2):
	t=0
	test = tanh(np.dot(X_test , weights1))
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
nn_no_bias = NeuralNetwork(X_train,y_train,0.1,False)
nn_with_bias = NeuralNetwork(X_train,y_train,0.1,True)
t1 = []
t2 = []

for i in range(1000):
	nn_no_bias.feedforward()
	nn_no_bias.backprop()
	nn_with_bias.feedforward()
	nn_with_bias.backprop()
	print(np.mean((nn_no_bias.y-nn_no_bias.output)**2),np.mean((nn_with_bias.y-nn_with_bias.output)**2))
	print(i)
	test, count1 = testing(X_train, y_train, nn_no_bias.weights1, nn_no_bias.weights2)
	t1.append(count1)
	test, count2 = testing(X_train, y_train, nn_with_bias.weights1, nn_with_bias.weights2)
	t2.append(count2)

test, count1 = testing(X_test, y_test, nn_no_bias.weights1, nn_no_bias.weights2)
test, count2 = testing(X_test, y_test, nn_with_bias.weights1, nn_with_bias.weights2)

print("Accuracy without bias = ", count1/40)
print("Accuracy with bias = ", count2/40)

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Accuracy with SVM (rbf Kernel)",clf.score(X_test,y_test))

#plotting 
plot_learning_curve(clf, X_train, y_train)
plt.title("training curve of SVM RBF kernel")
plt.show()

plt.gca()
plt.plot(np.asarray(t1)/160)
plt.title("training curve of the given neral network without bias")
plt.show()

plt.gca()
plt.title("training curve of the given neral network with bias")
plt.plot(np.asarray(t2)/160)
plt.show()