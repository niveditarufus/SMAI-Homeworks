import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier



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

train_data, train_labels = read_data("mnist_train.csv")
test_data, test_labels = read_data("mnist_test.csv")
train_data = train_data/255.0
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)
clf = OneVsOneClassifier(LogisticRegression(random_state=0))
clf.fit(train_data,train_labels)
predictions = clf.predict(test_data)
score = clf.score(test_data,test_labels)
print(score)



cm = metrics.confusion_matrix(test_labels, predictions)
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()