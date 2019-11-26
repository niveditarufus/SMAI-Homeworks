import numpy
import matplotlib.pyplot as plt


x1=numpy.random.uniform(-1.0,1.0,100)
x2=numpy.random.uniform(-1.0,1.0,100)

x1 = numpy.asarray(x1)
x2 = numpy.asarray(x2)

w=[[1,1,0],[-1,-1,0],[0.,5,0],[1,-1,5],[1,1,0.3]]
w=numpy.asarray(w)
fig, ax = plt.subplots(3)
i=0

for x in xrange(len(w)):
	sumA=0
	sumB=0
	A=[]
	B=[]
	for y in xrange(100):
		temp = numpy.dot(w[x],[x1[y],x2[y],1])
		if (temp > 0):
			if(y<50):
				sumA +=1
			A.append([x1[y],x2[y],1])

		else:
			if(y>=50):
				sumB += 1
			B.append([x1[y],x1[y],1])

	print "w = ",w[x]
	print "Percentage accuracy : ",sumA+sumB
	A=numpy.asarray(A)
	B=numpy.asarray(B)
	B=numpy.reshape(B,(len(B),3))
	
	
	if(x==0 or x==3 or x==4):
		ax[i].scatter(A[:,0],A[:,1], marker='^', label="class A")
		ax[i].scatter(B[:,0],B[:,1], c='green', marker='o',label="class B")
		i=i+1
x=numpy.linspace(-1,1,10)
y=-x
ax[0].plot(x, y, '-r', label='w=[1,1,0]')
ax[0].legend(loc='upper left')

y = x-5
ax[1].plot(x, y, '-r', label='w=[1,-1,5]')
ax[1].legend(loc='upper left')

y=-x-0.3
ax[2].plot(x, y, '-r', label='w=[1,1,0.3]')
ax[2].legend(loc='upper left')

plt.show()

