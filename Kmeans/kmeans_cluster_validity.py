import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import davies_bouldin_score 



x1 = np.array([[0,0],[0,1],[1,0],[-1,0],[0,-1],[0,0.5],[0,-0.5],[0.5,0],[-0.5,0],[0.5,0.5]])
x2 = np.array([[6,6],[6,7],[7,6],[5,6],[5,7],[6,5],[7,5],[6.5,6],[5.5,6],[6.5,7]])

X = np.vstack((x1,x2))
# print(X.shape)
plt.scatter(x1[:,0],x1[:,1], label = 'cluster 1')
plt.scatter(x2[:,0],x2[:,1],label = 'cluster 2')
plt.title("clustering")
plt.legend(loc = "upper left")
plt.show()
di = []
K = np.arange(2,10)

for k in K:
	clus = []
	kmeans = KMeans(n_clusters= k, init = 'random').fit(X) 
	labels = kmeans.predict(X)
	# print(davies_bouldin_score(X, labels))
	di.append(davies_bouldin_score(X, labels)) 
plt.plot(K,di)
plt.title("davies_bouldin_score vs K") 
plt.show()
