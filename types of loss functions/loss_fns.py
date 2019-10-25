import numpy as np 
import matplotlib.pyplot as plt

x = -2
y = x**2

w = 2
w1 = w
w2 = w
w3 = w
n = 0.1
n2 = 1
n3 = -0.0007
e = []
l = []
l1 =[]
l2 =[]
l3=[]
for i in range(30):
	# print(w1*x)
	# error = (y - w*x)**2
	l1.append(w1*x)
	l2.append(w2*x)
	l3.append(w3*x)

	# w = w + (n*x*(y - w*x))
	w1 = w1 + (n*x*(y - w1*x))
	w2 = w2 + (n2*x*(y - w2*x))
	w3 = w3 + (n3*x*(y - w3*x))

	# plt.plot(i,error)
	# print(error,w)
	# e.append(error)
# print(l)
# l = np.asarray(l)
# plt.plot(l[:,0],l[:,1])
# plt.plot(l)
# print(len(l1))
print(l1,l2,l3)
plt.plot(l1,'r',label = 'convergence n = 0.1')
plt.plot(l2,'b',label='oscilation n = 1')
plt.plot(l3,'k',label='divergence n= -0.1')
# plt.plot(e)
# plt.title('convergence')
plt.ylim((6,-6))
plt.legend()
plt.show()