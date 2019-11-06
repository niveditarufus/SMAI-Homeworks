import numpy as np 
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


alpha = np.array([1,-0.0021,0.5,0.01,0.041])
data = np.zeros([50])
data[0] = 0.2
for i in range(1,5):
	data[i] = data[i-1]*alpha[i] + data[i] + data[i-1] + np.random.normal(0,10)
for i in range(5,50):
	data[i] = np.matmul(alpha.T,data[i-5:i]) + np.random.normal(0,10)
test = np.zeros([50])
test[0] = 0.2
for i in range(1,5):
	test[i] = test[i-1]*alpha[i] + test[i] + test[i-1] + np.random.normal(0,10)
for i in range(5,50):
		test[i] = np.matmul(alpha.T,test[i-5:i]) + np.random.normal(0,10)


#plotting data
plt.plot((data),'r')
plt.title("data")
plt.show()


#one step(training)
y = data[1:]
x = data[0:49]

x = np.reshape(x,(-1,1))
y = y.ravel()
clf = LinearRegression()
clf.fit(x,y) 
#one step(Testing)
test_y = test[1:]
test_x = test[0:49]
test_x = np.reshape(test_x,(-1,1))
test_y = test_y.ravel()
predictions = clf.predict(test_x)
print("test accuracy : ",r2_score(test_y,predictions))
plt.plot((predictions),label="predicted")
plt.plot((test_y),label="actual")
plt.legend()
plt.xlabel("time(months)")
plt.ylabel("price")
plt.show()


#E vs d

d = 20
error = []
step = []
for i in range (1,d+1):
	print(i)

	y = data[i:]
	x = data[0:50-i]
	x = np.reshape(x,(-1,1))
	y = y.ravel()

	clf = LinearRegression()
	clf.fit(x,y) 
	#testing with different values of d
	test_y = test[i:]
	test_x = test[0:50-i]
	test_x = np.reshape(test_x,(-1,1))
	test_y = test_y.ravel()
	predictions = clf.predict(test_x)
	print("test accuracy : ",r2_score(test_y,predictions))
	error.append(r2_score(test_y,predictions))
	step.append(i)
plt.plot(np.asarray(step).ravel(),1-np.asarray(error).ravel(),'r-')
plt.xlabel("d")
plt.ylabel("error")
plt.show()