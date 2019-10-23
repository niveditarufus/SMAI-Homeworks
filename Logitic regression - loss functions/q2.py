import numpy as np
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

x1 = np.random.normal(1,2,1000)
x2 = -np.random.normal(2,2,1000)
x = np.hstack((x1,x2))
x = np.ravel(x)
x = np.sort(x)[::-1]
def sigmoid(x):
	return 1/(1+np.exp(-x))
w = np.arange(-5,5,0.5)
bias = -1.5
log_loss = []
temp =[]
# mse =[]
for i in range(10):
	temp1 = w[i]*x + bias
	temp.append(temp1)

	temp1 = 1/sigmoid(temp1)
	log_loss.append(np.log(temp1))
ax1.plot(temp,log_loss,'r-')
ax1.set_title('negative log likelihood')

temp =[]
loss =[]
for i in range(10):
	temp1 = w[i]*x + bias
	temp.append(temp1)
	temp1 = -sigmoid(temp1)
	mse = np.where(x<0,-1+temp1,1+temp1)
	loss.append(mse.tolist())
print(mse.shape)
ax2.plot(temp,loss,'b-')
ax2.set_title('MSE loss')

plt.show()
