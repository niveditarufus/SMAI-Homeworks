import numpy as np 
import matplotlib.pyplot as plt
x = np.linspace(0,10,1000)
# x = x/10
y  = x
noise = np.random.normal(0,1,1000)
x = x+noise
gamma = 0.25
n = 0.1
w1 = 2
w2 =w1
w3 =w1
l1 =[]
l2 =[]
l3 =[]
coeff1 = []
coeff2 =[]
coeff3 = []
# plt.plot(x,'r.')
# plt.title("data")
def loss1(true,pred):
	mae_loss = np.mean(np.abs(true - pred))
	return mae_loss

def loss2(true,pred):
	logcosh_loss = np.sum(np.log(np.cosh(pred - true)+1))
	return logcosh_loss/1000

def loss3(true,pred,gamma=0.25):
	quan_loss = np.sum(np.where(pred>true , (gamma-1)*abs(pred-true), gamma*np.abs(true - pred)))
	return quan_loss/1000


def function1(w,x,y,n):
	temp = np.multiply(x,(y - (w*x))/abs(y-(w*x)))
	a = w + (n*np.sum(temp)/1000)
	return a

def function2(w,x,y,n):
	a = w - ((n*np.sum(np.multiply(x,np.tanh(w*x - y))))/1000)
	return a

def function3(w,x,y,n,gamma=0.25):
	a = w - (n*np.sum((x)*(2*gamma - 1)*(w*x - y)/abs(w*x -y))/1000)
	return a

mae_loss = loss1(y,w1*x)
logcosh_loss = loss2(y,w2*x)
quan_loss = loss3(y,w3*x)
for i in range(100):
	mae_loss = loss1(y,w1*x)
	logcosh_loss = loss2(y,w2*x)
	quan_loss = loss3(y,w3*x)
	# print(mae_loss,logcosh_loss,quan_loss)
	print(w1,w2,w3)
	# print(y,w1*x)
	l1.append(mae_loss)
	l2.append(logcosh_loss)
	l3.append(quan_loss)
	coeff1.append(w1)
	coeff2.append(w2)
	coeff3.append(w3)
	w1 = function1(w1,x,y,n)
	# print(w1)
	w2 = function2(w2,x,y,n)
	w3 = function3(w3,x,y,n)

# l1=np.asarray(l1)
# l2=np.asarray(l2)	
# l3=np.asarray(l3)
# print(coeff1)
coeff1=np.asarray(coeff1)
coeff2=np.asarray(coeff2)	
coeff3=np.asarray(coeff3)
# plt.plot(l1,label = 'max absolute error')
# plt.plot(l2,label = "log cosh error")
# plt.plot(l3,label = 'quantile error')
plt.plot(coeff1,label = 'max absolute loss')
plt.plot(coeff2,label = "log cosh ")
plt.plot(coeff3 ,label = 'quantile ')
plt.title("w vs time")
plt.legend()
# plt.plot(l2[:,0],l2[:,1])
# plt.plot(l3[:,0],l3[:,1])
plt.ylim((0,2))


plt.show()