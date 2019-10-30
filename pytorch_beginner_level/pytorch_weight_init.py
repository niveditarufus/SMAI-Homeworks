#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
# fig,ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))


# In[2]:


class MNIST(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X.index)
    
    def __getitem__(self, index):
        image = self.X.iloc[index, ].values.astype(np.uint8).reshape((28, 28, 1))
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.y is not None:
            return image, self.y.iloc[index]
        else:
            return image


# In[3]:


train_df = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

print('train data shape : ', train_df.shape)
print('test data shape : ',X_test.shape)


# In[4]:


X_train, X_valid, y_train, y_valid = train_test_split(train_df.iloc[:, 1:], train_df['label'], test_size=1/6, random_state=42)


# In[5]:


transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

train_dataset = MNIST(X=X_train, y=y_train, transform=transform)
valid_dataset = MNIST(X=X_valid, y=y_valid, transform=transforms.ToTensor())
test_dataset = MNIST(X=X_test, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


# In[6]:

def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0,1.0)
        m.bias.data.fill_(0)

def init_weights_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        self.layers.apply(init_weights_normal)
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


# In[7]:


model = MLP()
loss_fn = nn.CrossEntropyLoss()


# In[9]:


mean_train_losses = []
epochs = 15
lr = np.array([0.0001])
for j in range (1):
    print(j)
    mean_train_losses = []
    mean_valid_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
    
        train_losses = []
        valid_losses = []
        for i, (images, labels) in enumerate(train_loader):
        
            optimizer.zero_grad()
        
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            train_losses.append(loss.item())
            mean_train_losses.append(np.mean(train_losses))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
           for i, (images, labels) in enumerate(valid_loader):
               outputs = model(images)
               loss = loss_fn(outputs, labels)
               
               valid_losses.append(loss.item())
               
               _, predicted = torch.max(outputs.data, 1)
               correct += (predicted == labels).sum().item()
               total += labels.size(0)
                   
           mean_train_losses.append(np.mean(train_losses))
           mean_valid_losses.append(np.mean(valid_losses))
           
           accuracy = 100*correct/total

    plt.plot(mean_train_losses, label='Train loss lr = '+str(lr[j]))
    plt.title('Normal')
    # plt.plot(mean_valid_losses, label='validation loss lr = '+str(lr[j]))
    plt.legend()
    


# In[10]:


plt.show()


# In[ ]:





# In[ ]:



