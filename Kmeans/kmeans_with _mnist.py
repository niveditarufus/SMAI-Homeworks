import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score 
from sklearn.cluster import KMeans 
from mlxtend.plotting import plot_decision_regions
from matplotlib import rcParams, cycler
from sklearn.metrics.cluster import homogeneity_score
from random import *
import numpy_indexed as npi




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

train_data, train_labels = read_data("mnist_train.csv")
test_data, test_labels = read_data("mnist_test.csv")

pca = PCA(n_components=2)
data = pca.fit_transform(train_data)  
print(data.shape)

kmeans = KMeans(n_clusters= 10,init = 'random').fit(data) 
labels = kmeans.predict(data)
plt.scatter(data[:,0], data[:,1],c = train_labels.astype(np.int),cmap = 'rainbow' )
# plot_decision_regions(X=data, y=labels.astype(np.int), clf=kmeans, legend=10)

# fig, ax = plt.subplots()
# lines = ax.scatter(data[:,0],data[:,1])
plt.title("Ground truth of MNIST")
plt.legend()

# plt.title("Kmeans clustering of MNIST")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.show()
f, ax = plt.subplots()
kmeans = KMeans(n_clusters= 10,init = 'random').fit(data) 
labels = kmeans.predict(data)
plot_decision_regions(X=data, y=labels.astype(np.int), clf=kmeans, legend=10)
plt.title("Random initialisation ,Score = "+str(homogeneity_score(train_labels, labels)))
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.legend(loc = 'upper right')
# 
plt.show()