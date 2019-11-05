import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm,metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,scale
from mlxtend.plotting import plot_decision_regions

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
def make_meshgrid(x, y, h=.2):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    ind = 0

    for line in (lines):
        num = line.split(',')

        if((int(num[0])==1) or (int(num[0])==2)):
            labels[ind] = int(num[0])
            data[ind] = [ int(x) for x in num[1:] ]
            ind = ind + 1
            # print(int(num[0]))

    data = data[:ind,:]
    labels = labels[:ind]
    return (data, labels)



train_data, train_labels = read_data("mnist_train.csv")
test_data, test_labels = read_data("mnist_test.csv")
print(train_labels.shape)


train_data = StandardScaler().fit_transform(train_data)
train_data = scale(train_data)

test_data = StandardScaler().fit_transform(test_data)
test_data = scale(test_data)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(train_data)
tpc = pca.fit_transform(test_data)

clf = svm.SVC(C =1,gamma=0.0001,kernel = 'linear')
clf.fit(principalComponents,train_labels)

predictions = clf.predict(tpc)
score = clf.score(tpc,test_labels)
print(score)

score = clf.score(principalComponents,train_labels)
print(score)
fig, ax = plt.subplots()
X0, X1 = principalComponents[:, 0], principalComponents[:, 1]
xx, yy = make_meshgrid(X0, X1)
print(clf.support_vectors_.shape)
plot_contours(ax, clf, xx, yy, cmap='PiYG', alpha=0.8)
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Decison surface using the PCA transformed features')
# ax.set_title("support_vectors for the first two principal principal Components")
# ax.set_title("decision boundary along with support_vectors")
plot_decision_regions(X=principalComponents, 
                      y=train_labels.astype(np.int),
                      clf=clf, 
                      legend=2)
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=50,
#            linewidth=1,c= 'k',marker = 'o', edgecolors='k',label = 'support_vectors')
ax.legend()

plt.show()