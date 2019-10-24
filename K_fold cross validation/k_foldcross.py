import numpy as np
import math
from random import randrange
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2)
def cross_validation_split(dataset, folds=10):
	fold_size = int(len(dataset) / folds)
	dataset_copy = list(dataset)
	dataset_split = []
	for i in range(folds):
		fold = []
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	dataset_split = np.asarray(dataset_split)
	return dataset_split

def get_train_test(dataset,i):
	copy = dataset
	test = dataset[i-1:i,:]
	train = np.delete(copy,i-1,axis =0)
	return test,train


# k = 10
N = np.random.normal(1,1,1000)
y = math.sin(0.0) + N
plotsig = []
plotmu = []

for k in range(2,100):
	K_datasets = cross_validation_split(y,k)
	# print(K_datasets.shape)
	mu = []
	sigma = []
	for i in range(1,int(1000/k)):
		test,train = get_train_test(K_datasets.T,i)
		# print(train.shape)
		mu_train = np.mean(train)
		sigma_train = np.std(train)**2
		mu_test = np.mean(test)
		sigma_test = np.std(test)
		mu.append(mu_train)
		sigma.append(sigma_train)
	m = sum(mu)/len(mu)
	s = sum(sigma)/len(sigma)
	plotmu.append([k,m])
	plotsig.append([k,s])
plotmu = np.asarray(plotmu)
plotsig = np.asarray(plotsig)
ax[0].plot(plotmu[:,0],plotmu[:,1],'ro-')
ax[0].set_ylim(0.5, 1.5)
ax[0].set_title("K vs mu")
ax[1].plot(plotsig[:,0],plotsig[:,1],'go-')
ax[1].set_ylim(0.5, 1.5)
ax[1].set_title("K vs sigma")
plt.show()



