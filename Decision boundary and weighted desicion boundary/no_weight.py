import numpy as np 
import math
import matplotlib.pyplot as plt
def ezplot(s):
    #Parse doesn't parse = sign so split
    lhs, rhs = s.replace("^","**").split("=")
    eqn_lhs = parse_expr(lhs)
    eqn_rhs = parse_expr(rhs)

    plot_implicit(eqn_lhs-eqn_rhs)

x1 = np.array([0,0,2,3,3,2,2])
y1 = np.array([0,1,0,2,3,2,0])
x1 = np.reshape(x1,(1,7))
y = np.reshape(y1,(1,7))
mx = np.mean(x1)
my = np.mean(y1)
m1 = np.array([mx,my])
m1 = np.reshape(m1,(2,1))
x1 = x1-mx
y1 = y1-my

x2 = np.array([7,8,9,8,7,8,7])
y2 = np.array([7,6,7,10,10,9,11])
x2 = np.reshape(x2,(1,7))
y2 = np.reshape(y2,(1,7))
mx = np.mean(x2)
my = np.mean(y2)
m2 = np.array([mx,my])
m2 = np.reshape(m2,(2,1))

cov1 = np.cov(x1,y1)
d1 = np.linalg.det(cov1)
d1 = math.sqrt(d1)
icov1 = np.linalg.inv(cov1)

cov2 = np.cov(x2,y2)
d2 = np.linalg.det(cov2)
d2 = math.sqrt(d2)
icov2 = np.linalg.inv(cov2)
w1 = (np.matmul(icov1,m1))
w10 = -0.5*(np.matmul(np.matmul(m1.T,icov1),m1)) + math.log(0.5)
w2 = (np.matmul(icov2,m2))
w20 = -0.5*(np.matmul(np.matmul(m2.T,icov2),m2)) + math.log(0.5)

d = [[0,0],[2,2],[4,4],[8,6],[8,8],[10,10],[3,1],[4,3],[5,1],[0,3.5],[10,7],[10,9],[11,11]]
for x in d:
	x = np.asarray(x)
	
	g1 = np.matmul(w1.T,x.T) + w10
	g2 = np.matmul(w2.T,x) + w20
	g = g1-g2
	

	if g>0:
		plt.scatter(x[0],x[1],color = 'r', marker = 'o')
	else:
		plt.scatter(x[0],x[1],color = 'g', marker = '^')
A = w1.T - w2.T

w0 = (w20 -w10) 
x = np.linspace(0,10,100)
y = (A[:,0]*x - w0)/A[:,1]
print(y.shape)

plt.plot(x,y.T,'b')

plt.show()