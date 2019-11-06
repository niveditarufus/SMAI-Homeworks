#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torchvision.datasets as datasets



# In[2]:


BATCH_SIZE = 4

preprocess = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

#Loading the train set file
dataset = datasets.MNIST(root='./data',
                            transform=preprocess,  
                            download=True)

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='./data',
                            transform=preprocess,  
                            download=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[3]:


def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


# In[4]:


class SequentialMNIST(nn.Module):
    def __init__(self):
        super(SequentialMNIST, self).__init__()
        self.linear1 = nn.Linear(28*28, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        h_relu = F.relu(self.linear1(x.view(BATCH_SIZE, -1)))
        y_pred = self.linear2(h_relu)
        return y_pred


# In[10]:


model = SequentialMNIST()
def train(model, trainloader, criterion, optimizer, n_epochs=2):
    for t in range(15):
        print(t)
        
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
train(model, loader, criterion, optimizer)


# In[28]:


def predict(model, images):
    outputs = model(Variable(images))
    _, predicted = torch.max(outputs.data, 1)  
    return predicted

def test(model, testloader, n, acc):
    correct = 0
    k=0
    for data in testloader:
        inputs, labels = data
        
        pred = predict(model, inputs)
        correct += (pred == labels).sum()
        
        for g in range(4):
            if pred[g] != labels[g] :
                acc.append(labels[g])
        
    plt.hist(acc,label = "no. of misclassifications with MLP")
    plt.xlabel("labels")
    plt.ylabel("Misclassified digits")
    plt.legend()
    plt.show()
        
    return 100 * correct / n
acc = []
test(model, test_loader, len(test_loader),acc)


# In[12]:


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
    
    def forward(self,x):
        h = self.encoder(x)
        return h
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
train(model, loader, criterion, optimizer)


# In[29]:


acc = []
test(model, test_loader, len(test_loader),acc)


# In[ ]:




