import numpy as np 
import matplotlib.pyplot as plt
x = np.random.normal(0,1,10000)
s =500
# s =[1,2,10,20,25,50,100,200,500,1000,5000]
# y = np.empty([100,1])
k=0
# k = 1000
mean_s=[]
variance =[]
klist=[]
slist =[]
# for i in range(s):
for i in range(s):
	y =[]
	print(i)
	k = 500 +k
	for j in range(k):
		y.append(x[np.random.randint(0,10000)])
	y =np.asarray(y)
	mean_s.append(np.mean(y))
	variance.append(np.var(y))
	klist.append(k)
	slist.append(i)
# plt.plot(slist,variance)
plt.plot(klist,variance)
# plt.title("variance vs k at s=10")
plt.title("variance vs k s =500")
print("estimates of mean :")
print(mean_s) 
plt.show()


