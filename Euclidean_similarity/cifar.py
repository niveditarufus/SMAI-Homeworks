import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import scipy


def one_hot_encoded(classnumbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(classnumbers) + 1
    return np.eye(num_classes, dtype=float)[classnumbers]

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file
def load_class_names():
    raw = _unpickle(filename="batches.meta")['label_names']
    return raw

def _get_file_path(filename=""):
    return os.path.join("/home/nive/space/smai/ass10", "cifar-10-batches-py/", filename)

def _unpickle(filename):
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file,  encoding='latin1')
    return data

def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def _load_data(filename):
    data = _unpickle(filename)
    raw_images = data['data']
    cls = np.array(data['labels'])
    images = _convert_images(raw_images)
    return images, cls

def load_training_data():
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[num_images_train], dtype=int)
    begin = 0
    for i in range(num_files_train):
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
    return images, cls, one_hot_encoded(classnumbers=cls, num_classes=num_classes)
def organise(images, cls,class_names):
	xorg = np.empty([10,5000,32,32,3])
	for i in range(10):
		temp = np.empty([5000,32,32,3])
		j=0
		# print(cls.shape[0])
		for x in range(cls.shape[0]):
			if(i==(cls[x])):
				temp[j] =images[x]
				# print(temp[j])
				j = j+1
		xorg[i] = temp
	return xorg
# def get_mean(mean_im,x):
# 	mean_im = mean_im.reshape(mean_im.shape[0],-1)
# 	pca = PCA(n_components=20, whiten=True)
# 	x_train_pca = pca.fit_transform(mean_im)
# 	eig_vals,vec = np.linalg.eig(np.cov(mean_im.T))
# 	eig_valspca,vecpca = np.linalg.eig(np.cov(x_train_pca.T))
# 	error = abs(np.sum(eig_vals) - np.sum(eig_valspca))
# 	return error
def get_index(t,i):
	idx = np.argpartition(t,4)
	print("class : "+str(i+1))
	print(idx[:4]+1)


class_names = load_class_names()
images_train, cls_train, labels_train = load_training_data()
xorg = organise(images_train,cls_train,class_names)
# print(xorg[9])
mean = []
dist = []
pair_dist = []
for i in xorg:
	mean_im = np.mean(i,axis =0)
	x = i.reshape(i.shape[0],-1)
	# print(x)
	# mean.append(get_mean(mean_im,x))
	dist.append(mean_im)
print("similarity between the classes:")

for i in range(len(dist)):
	t=[]
	# print("distance of all classes from class"+str(i+1)+" :")

	for j in range(len(dist)):
		e = scipy.spatial.distance.euclidean(dist[i].ravel(),dist[j].ravel())
		t.append(e)
	get_index(np.array(t),i)
	pair_dist.append(np.array(t))
pair_dist = np.asarray(pair_dist)
# print("Euclidean distance matrix:")
# print(pair_dist)
# 		print(("class "+str(j+1),e))
