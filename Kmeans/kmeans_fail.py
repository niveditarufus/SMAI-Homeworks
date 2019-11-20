from sklearn.metrics import pairwise_distances_argmin
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


x = [-1,-1,0,0,1,1,0,0]
y = [1,-1,0.5,-0.5,1,-1,1,-1]

X = np.asarray(list(zip(x, y)))

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters, here let it be:
    centers = [[-0.5, 0],[ 0.5, 0]]
    itercounter = 0
    while True:
        itercounter+=1
        # 2a. Assign labels based on closest center
        # the below determistic assignment is valid because
        # the middle 4 datapoints can belong to either classes
        if itercounter%2 == 0:
            oldlabels = np.asarray([1,1,1,1,0,0,0,0])
        else:
            oldlabels =  np.asarray([1,1,0,0,0,0,1,1])
        print(oldlabels)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[oldlabels == i].mean(0)
                                for i in range(n_clusters)])

        print(new_centers)
        
        newlabels = pairwise_distances_argmin(X, new_centers)
        print(newlabels)
        
        # 2c. Check for convergence 
        # one is based on centers one is based on labels not changing
        if np.all(oldlabels == newlabels):
            break
        # if np.all(centers == new_centers):
        #     break
        centers = new_centers
    
    return centers, newlabels, itercounter

centers, labels, niter= find_clusters(X, 2)
print(niter)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');
plt.show()