import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


# x = np.arange(0,50)
# X = np.vstack((x,x))

# X = X.T
X = np.array([[0,3],[1,3],[10,20],[11,20],[100,50],[101,50]])
print(X.shape)
plt.scatter(X[:, 0], X[:, 1], s=50);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# x <- y <- c(rep(0,3),rep(1,3),rep(10,20),rep(11,20),rep(100,50),rep(101,50))
# m <- cbind(x,y)

# plot(m, cex=4, lwd=2)