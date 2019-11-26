import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial import distance
import math
def shortest_distance(x1, y1, a, b=1):  
       
    d = abs((-a * x1 + b * y1 )) / (math.sqrt(a * a + b * b))
    return d 
      

mean1 = [1,1]
cov1 = [[2,1],[1,3]]
mean2 = [3,8]
cov2 = [[2,1],[1,3]]

x1 = np.random.multivariate_normal(mean1, cov1, 500)
x2 = np.random.multivariate_normal(mean2, cov2, 500)


for j in xrange(1):
	print(j)
	y1 = []
	y2 = []

	C1 = np.cov(x1.T)
	C2 = np.cov(x2.T)
	eigenValues, eigenVectors = np.linalg.eig(C1)
	idx = eigenValues.argsort()[::-1]   
	e1 = eigenValues[idx][0]
	v1 = eigenVectors[:,idx][0]
	m1 = v1[1]/v1[0]

	eigenValues, eigenVectors = np.linalg.eig(C2)
	idx = eigenValues.argsort()[::-1]   
	e2 = eigenValues[idx][0]
	v2 = eigenVectors[:,idx][0]
	m2 = v2[1]/v2[0]

	a = np.linspace(-5,5,30)
	b1 = m1*a
	b2 = m2*a
	
	for i in range(len(x1)):
		d1 = shortest_distance(x1[i][0], x1[i][1],m1)
		d2 = shortest_distance(x1[i][0], x1[i][1],m2)
		if d1<d2:
			y1.append(x1[i])
		else:
			y2.append(x1[i])

	for i in range(len(x2)):
		d1 = shortest_distance(x2[i][0], x2[i][1],m1)
		d2 = shortest_distance(x2[i][0], x2[i][1],m2)
		if d1<d2:
			y1.append(x1[i])
		else:
			y2.append(x1[i])

	y1 = np.asarray(y1)
	y2 = np.asarray(y2)

	print(y1.shape, y2.shape)

	if(np.array_equal(x1,y1)):
		
		break
	else:
		x1 = y1
		x2 = y2

plt.plot(a,b1,label = "Eigen vector of cluster 1 after convergence")
plt.plot(a,b2,label = "Eigen vector of cluster 2 after convergence")
plt.scatter(x1[:, 0], x1[:, 1], c= 'b',marker = '^',label = "cluster 1 after convergence")
plt.scatter(x2[:, 0], x2[:, 1], c= 'g', label = "cluster 2 after convergence")
plt.legend()
plt.show()