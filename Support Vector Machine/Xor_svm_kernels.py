import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
# fig, axs = plt.subplots(3,3)


def decision_regions(X, y, classifier,C,test_idx=None, resolution=0.02):
   markers = ( 'o', '^')
   colors = ('black', 'blue')
   cmap='PiYG'

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.6, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())


   X_test, y_test = X[test_idx, :], y[test_idx]
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               alpha=0.8, c=colors[idx],
               marker=markers[idx], label=cl)




x = np.array([[1,1],[-1,-1],[1,-1,],[-1,1]])
y = np.array([1,1,-1,-1])
x = np.vstack((x,(x+0.1)))
y = np.hstack((y,y))

C = np.array([0.01,2,1000])

plt.subplot(3,3,1)
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=C[0])
svm.fit(x, y)
decision_regions(x, y, svm,C[0])
plt.title("RBF kernel, C = "+str(C[0]))
plt.legend()

plt.subplot(3,3,2)
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=C[1])
svm.fit(x, y)
decision_regions(x, y, svm,C[1])
plt.title("RBF kernel, C = "+str(C[1]))
plt.legend()

plt.subplot(3,3,3)
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=C[2])
svm.fit(x, y)
decision_regions(x, y, svm,C[2])
plt.title("RBF kernel, C = "+str(C[2]))
plt.legend()


plt.subplot(3,3,4)
svm = SVC(kernel='poly', degree = 2,random_state=0, gamma=0.010, C=C[0])
svm.fit(x, y)
decision_regions(x, y, svm,C[0])
plt.title("Poly kernel, C = "+str(C[0]))
plt.legend()

plt.subplot(3,3,5)
svm = SVC(kernel='poly', degree = 3,random_state=0, gamma=0.010, C=C[1])
svm.fit(x, y)
decision_regions(x, y, svm,C[1])
plt.title("Poly kernel, C = "+str(C[1]))
plt.legend()

plt.subplot(3,3,6)
svm = SVC(kernel='poly', degree = 5,random_state=0, gamma=0.010, C=C[2])
svm.fit(x, y)
decision_regions(x, y, svm,C[2])
plt.title("Poly kernel, C = "+str(C[2]))
plt.legend()

plt.subplot(3,3,7)
svm = SVC(kernel='sigmoid', coef0 = 0.5,random_state=0, gamma=0.10, C=C[0])
svm.fit(x, y)
decision_regions(x, y, svm,C[0])
plt.title("sigmoid kernel, C = "+str(C[0]))
plt.legend()

plt.subplot(3,3,8)
svm = SVC(kernel='sigmoid', coef0 = 0.5,random_state=0, gamma=0.10, C=C[1])
svm.fit(x, y)
decision_regions(x, y, svm,C[1])
plt.title("sigmoid kernel, C = "+str(C[1]))
plt.legend()

plt.subplot(3,3,9)
svm = SVC(kernel='sigmoid', coef0 = 0.5,random_state=0, gamma=0.10, C=C[2])
svm.fit(x, y)
decision_regions(x, y, svm,C[2])
plt.title("sigmoid kernel, C = "+str(C[2]))
plt.legend()


plt.show()


# for j in range(1,4):
#   if(j==1):
#     for i in range(3):

#       print(i,j)
#       svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=C[i])
#       svm.fit(x, y)
#       plot_decision_regions(x, y, svm,C[i])
#       plt.title("RBF kernel, C = "+str(C[i]))


#   elif(j==2):
#     for i in range(3):
#       k= j+i
#       plt.subplot(3,3,k)

#       print(i,j)

#       svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=C[i])
#       svm.fit(x, y)
#       plot_decision_regions(x, y, svm,C[i])

#   else:
#       for i in range(3):
#         plt.subplot(3,3,j+i)

#         print(i,j)

#         svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=C[i])
#         svm.fit(x, y)
#         plot_decision_regions(x, y, svm,C[i])

# plt.show()