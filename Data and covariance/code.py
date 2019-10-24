import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import multivariate_normal



# mean1 = [3,3]
# mean2 = [7,7]
# cov1 = [[3,1],[2,3]]
# cov2 = [[7,2],[1,7]]
# #Create grid and multivariate normal
# x1 = np.linspace(0,10,1000)
# y1 = np.linspace(0,10,1000)
# X1, Y1 = np.meshgrid(x1,y1)
# pos = np.empty(X1.shape + (2,))
# pos[:, :, 0] = X1; pos[:, :, 1] = Y1
# rv = multivariate_normal(mean1, cov1, 1000)



# #Make a 3D plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # ax.plot(rv,'r.',linewidth=0)

# x2 = np.linspace(0,10,1000)
# y2 = np.linspace(0,10,1000)
# X2, Y2 = np.meshgrid(x2,y2)
# pos = np.empty(X2.shape + (2,))
# pos[:, :, 0] = X2; pos[:, :, 1] = Y2
# rv = multivariate_normal(mean2, cov2, 1000)
# print(rv)
# ax.plot(X2, Y2,'g.',linewidth=0)



# plt.show()
mean1 = [3,3]
# cov1 = 3*np.identity(2)
cov1 = [[3,1],[2,3]]
cov2 = [[7,2],[1,7]]
mean2 = [7,7]

# cov2 = cov1
# cov2 = [[4, 1, -1], [1, 2,1], [-1,1, 2]]

x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T


fig= plt.figure()
# ax = fig.add_subplot(111,projection='3d')
ax= fig.add_subplot(111)
# ax.scatter(x,y,z)
# ax.scatter(x1,y1,z1, c='r',marker ='o')
# ax.scatter(x2,y2,z2, c='b',marker ='+')
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=300, lw=1, facecolors='none');
ax.plot(x1,y1, 'r.')
ax.plot(x2,y2, 'g.')
np.linspace(0,10,1000)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_xlim(0,10)
ax.set_ylim(0,10)
# ax.plot(y, z, 'g+', zdir='x')
# ax.plot(x, y, 'k+', zdir='z')
# ax.set_xlabel("y - axis")
# ax.set_ylabel("z - axis")
# ax.set_zlabel("Z - axis")

ax.set_title("With differnt mean and PSD covariance matrix yz plane NOTE: scale on axes are different")
plt.show()