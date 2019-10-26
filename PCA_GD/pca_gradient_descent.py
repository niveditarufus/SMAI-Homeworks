import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

def normal_pca(X,k=2):
    X -= np.mean(X,axis = 0)
    C = np.cov(X.T)
    eig_vals, eigen_vec = np.linalg.eig(C)
    eigen_vec = eigen_vec[:,:k]
    Xnew = eigen_vec.T.dot(X.T)
    Xnew[0,:] *= -1 
    Xnew = Xnew.T
    print(Xnew.shape) 
    # plt.plot(Xnew[:,0],Xnew[:,1],'.',lw = 0.5)
    # plt.xlabel("vector1")
    # plt.ylabel("vector2")
    # plt.title("Normal PCA")
    return eigen_vec

def gradient_descent_pca(X,initial_guess,k=2,n=1):
    C = np.cov(X.T)
    v = initial_guess
    print(v.shape)
    for i in range(k):
        for j in range(100):
            v[:,i] = v[:,i] + n*(X.T.dot(X.dot(v[:,i])))
            v[:,i] /= np.linalg.norm(v[:,i])
        print(v[:,i])

    Xnew = v.T.dot(X.T)
    Xnew[0,:] *= -1 

    Xnew = Xnew.T
    print(Xnew.shape)

    plt.plot(Xnew[:,0],Xnew[:,1],'.',lw = 0.5)
    plt.xlabel("vector1")
    plt.ylabel("vector2")
    plt.title("Gradient descent version PCA")




train_data, train_labels = read_data("mnist_train.csv")
print(train_data.shape, train_labels.shape)
a = normal_pca(train_data)
initial_guess = 0.4*np.ones([784,2])/np.sqrt(784)
# initial_guess = a
gradient_descent_pca(train_data,initial_guess)
plt.show()
