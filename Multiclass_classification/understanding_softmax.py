import numpy as np
import matplotlib.pyplot as plt

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


def initialize(num_inputs,num_classes):
    w = np.random.randn(num_classes, num_inputs) / np.sqrt(num_classes*num_inputs) 
    b = np.random.randn(num_classes, 1) / np.sqrt(num_classes)
    return w,b

def softmax(z):
    exp_list = np.exp(z)
    result = 1/sum(exp_list) * exp_list
    result = result.reshape((len(z),1))
    assert (result.shape == (len(z),1))
    return result

def neg_log_loss(pred, label):
    loss = -np.log(pred[int(label)])
    return loss

def batch_gradient(w,b, x_batch, y_batch):
    
    batch_size = x_batch.shape[0]
    w_grad_list = []
    b_grad_list = []
    batch_loss = 0
    for i in range(batch_size):
        x,y = x_batch[i],y_batch[i]
        x = x.reshape((784,1)) # x: (784,1)
        E = np.zeros((10,1)) #(10*1)
        E[int(y)][0] = 1 
        pred = softmax(np.matmul(w, x)+b) #(10*1)

        loss = neg_log_loss(pred, y)
        batch_loss += loss

        w_grad = E - pred
        w_grad = - np.matmul(w_grad, x.reshape((1,784)))
        w_grad_list.append(w_grad)

        b_grad = -(E - pred)
        b_grad_list.append(b_grad)

    dw = sum(w_grad_list)/batch_size
    db = sum(b_grad_list)/batch_size
    return dw, db, batch_loss

def train(w,b,alpha , x_train, y_train, x_test, y_test):
    
    batch_size = 50
    learning_rate = alpha
    test_loss_list, test_accu_list = [],[]

    for epoch in range(1000):
        
        rand_indices = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
        num_batch = int(x_train.shape[0]/batch_size)
        batch_loss100 = 0
        for batch in range(num_batch):
            index = rand_indices[batch_size*batch:batch_size*(batch+1)]
            x_batch = x_train[index]
            y_batch = y_train[index]

            dw, db, batch_loss = batch_gradient(w,b, x_batch, y_batch)
            batch_loss100 += batch_loss
            w -= learning_rate * dw
            b -= learning_rate * db
        print(batch_loss100)


train_data, train_labels = read_data("mnist_train.csv")
test_data, test_labels = read_data("mnist_test.csv")
train_data = train_data/255.0
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)
alpha = 0.1
num_inputs = train_data.shape[1]
num_classes = len(set(train_labels))
w,b = initialize(num_inputs,num_classes)
train(w, b, alpha , train_data, train_labels, test_data, test_labels)