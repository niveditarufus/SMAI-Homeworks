import numpy as np
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else -1

def train_weights(train, l_rate, n_epoch,weights):
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			print(error,prediction,row[-1])

			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):

				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
			print(weights)

		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights
l_rate = 1
n_epoch = 6
# dataset = [[1,1,1],[-1,-1,-1],[2,2,1],[-2,-2,-1],[-1,1,1],[1,-1,1]]
dataset = [[2,3,1],[3,2,1],[3,5,1],[0,0,-1],[1,2,-1],[2,0,-1]]

weights = [1,1,1]
weights = train_weights(dataset, l_rate, n_epoch,weights)
print(weights)
 
# test predictions

