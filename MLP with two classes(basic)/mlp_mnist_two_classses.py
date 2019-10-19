import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler,scale
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
import time

def plot_learning_curve(estimator,  X, y, ylim=None, cv=None,
                        n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, '-', color="r",
             label="score")

    plt.legend()
    return plt

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

train_data = StandardScaler().fit_transform(train_data)
train_data = scale(train_data)

test_data = StandardScaler().fit_transform(test_data)
test_data = scale(test_data)

start = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, alpha=1e-3,solver='sgd', verbose=10, tol=1e-3, random_state=1)
mlp.fit(train_data, train_labels)
print("Training set score: %f" % mlp.score(train_data, train_labels))
print("Test set score: %f" % mlp.score(test_data, test_labels))


# plot_learning_curve(mlp, train_data, train_labels)
# plt.title("training curve")
# plt.show()
end = time.time()
print("Time: ",end-start)