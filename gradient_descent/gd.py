import numpy as np
import random
import matplotlib.pyplot as plt

def read_data():
	a = np.loadtxt("wine.txt",delimiter = ',')
	x = a[:,1:]
	y = a[:,0]
	return a
def standardize(x):
	for i in range(x.shape[1]):
		x[:,i] = (x[:,i] - np.min(x[:,i]))/(np.max(x[:,i])-np.min(x[:,1]))
	return x
def create_mini_batches(data, batch_size): 
    mini_batches = [] 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        mini_batches.append((X_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        mini_batches.append((X_mini)) 
    return mini_batches 

def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat
 
def coefficients_sgd(train, l_rate, n_epoch):
	ploterror = []
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			sum_error += (abs(error))
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
		# print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		ploterror.append(sum_error)
	return coef,ploterror

def minibatch(train,learning_rate,num_epochs):
	ploterror = []
	for epoch in range(num_epochs):
		sum_error = 0
		for batch in create_mini_batches(train, batch_size=8):
			for row in batch:
				yhat = predict(row, coef)
				error = yhat - row[-1]
				sum_error += abs(error)
				coef[0] = coef[0] - l_rate * error
				for i in range(len(row)-1):
					coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
		# print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		ploterror.append(sum_error)
	return coef,ploterror

def gradientDescent(X, y, alpha, iters):
    ploterror = []
    theta = np.empty([13,1])
    temp = np.matrix(np.zeros(theta.shape))
    
    parameters = int(theta.ravel().shape[0])
    y = np.reshape(100,1)
    
    for i in range(iters):
        error = (np.matmul(X,theta) - y)
        ploterror.append(abs(np.sum(error)))
        # error = np.multiply(error,error)
        ploterror.append(abs(np.sum(error))/100)
        term = np.matmul(X.T,error)
        theta = theta - ((alpha / len(X)) * (term))
            
        # theta = temp
        
    return theta,ploterror
def testing(testdata, coeff):
	count =0
	for e in testdata:
		r = (np.matmul(coeff,e[1:].T))
		y = abs(e[0] - r)
		r = e[np.where(y == np.amin(y))]
		# print(r,e[0])
		if r != e [0]:
			count = count+1
	print(count)

 
x = read_data()

x = standardize(x)
np.random.shuffle(x)
train_data = x[:100,:]

test_data = x[100:,:]
l_rate = 0.01
n_epoch = 100
coef,sgd = coefficients_sgd(train_data[:,1:], l_rate, n_epoch)
# print(coef)
# coef = np.asarray(coef)
# coef = np.reshape(coef,(1,13))
# print(coef.shape)

testing(train_data,coef)

coef,mgd = minibatch(train_data[:,1:], l_rate, n_epoch)
# print(coef)
# print(x.shape,train_data.shape)
# coef = np.asarray(coef)
# coef = np.reshape(coef,(1,13))
testing(train_data,coef)
xn = np.empty([train_data.shape[0],train_data.shape[1]+1])
xn[:,0] = np.ones(train_data.shape[0])
xn = train_data
coef = np.empty([1,13])
# print(train_data.shape)
coef,bgd = gradientDescent(xn[:,1:],train_data[:,0], l_rate, n_epoch)
# print(coef)
# coef = list(coef)
testing(train_data,coef.T)


