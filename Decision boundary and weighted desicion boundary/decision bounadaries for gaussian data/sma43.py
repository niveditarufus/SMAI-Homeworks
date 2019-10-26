import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

mean1 = [20, 50, 50]
cov1 = [[10, 0, 0], [0, 10,0], [0, 0, 10]]
cov = [4, 1, -1], [1, 2,1], [-1,1, 2]
mean2 = [20, 50, 50]
cov2 = cov


x1, y1, z1 = np.random.multivariate_normal(mean1, cov1, 1000).T
x2, y2, z2 = np.random.multivariate_normal(mean2, cov2, 1000).T

fig= plt.figure()
ax= fig.add_subplot(111)
# ax.scatter(x,y,z)
# ax.scatter(x1,y1,z1, c='r',marker ='o')
# ax.scatter(x2,y2,z2, c='b',marker ='+')
ax.plot(z1,y1, 'ro')
ax.plot(z2,y2, 'b+')
# ax.plot(y, z, 'g+', zdir='x')
# ax.plot(x, y, 'k+', zdir='z')
ax.set_xlabel("z - axis")
ax.set_ylabel("y - axis")
# ax.set_zlabel("Z - axis")
ax.set_title("With same mean and different covariance matrix zy plane NOTE: scale on axes are different")
# ax.set_aspect('equal')
plt.show()