#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code here
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init as weight_init
import matplotlib.pyplot as plt
import pdb


#parameters
batch_size = 128

preprocess = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

#Loading the train set file
dataset = datasets.MNIST(root='./data',
                            transform=preprocess,  
                            download=True)

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


class AE(nn.Module):
    def __init__(self,dim1,dim2,dim3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, dim1),
            nn.ReLU(),
            nn.Linear(dim1,dim2),
            nn.ReLU(),
            nn.Linear(dim2,dim3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim3, dim2),
            nn.ReLU(),
            nn.Linear(dim2,dim1),
            nn.ReLU(),
            nn.Linear(dim1, 28*28),
            nn.Tanh()
        )
    
    def forward(self,x):
        h = self.encoder(x)
        xr = self.decoder(h)
        return xr,h


# In[39]:


import numpy as np
dim = np.array([[240,55,2],[250,60,2],[256,64,2]])
prod = np.prod(dim,axis=1)

reconstruction_error = []
for j in range(3):
    print(j)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(dim[j,0], dim[j,1], dim[j,2])
    net = AE(dim[j,0], dim[j,1], dim[j,2])
    net = net.to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-2
    weight_decay = 1e-5
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = 5

    epochLoss = []
    for epoch in range(num_epochs):
        total_loss, cntr = 0, 0
    
        for i,(images,_) in enumerate(loader):
        
                images = images.view(-1, 28*28)
                images = images.to(device)
        
                optimizer.zero_grad()
        
                outputs, _ = net(images)
        
                loss = criterion(outputs, images)
        
                loss.backward()
        
                optimizer.step()
       
                total_loss += loss.item()
                cntr += 1
    
        print ('Epoch [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, total_loss/cntr))
        epochLoss.append(total_loss/cntr)
    net = net.to("cpu")
    torch.save(net.state_dict(),'ae_model.ckpt')
    net = AE(dim[j,0], dim[j,1], dim[j,2])
    checkpoint = torch.load('ae_model.ckpt')
    net.load_state_dict(checkpoint)
    net = net.to(device)
    ndata = len(dataset)
    hSize = 2

    test_dataset = datasets.MNIST(root='./data',
                            transform=preprocess,  
                            download=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    iMat = torch.zeros((ndata,28*28))
    rMat = torch.zeros((ndata,28*28))
    featMat = torch.zeros((ndata,hSize))
    labelMat = torch.zeros((ndata))
    cntr=0

    with torch.no_grad():
        for i,(images,labels) in enumerate(loader):

            images = images.view(-1, 28*28)
            images = images.to(device)
        
            rImg, hFeats = net(images)
        
            iMat[cntr:cntr+batch_size,:] = images
            rMat[cntr:cntr+batch_size,:] = (rImg+0.1307)*0.3081
        
            featMat[cntr:cntr+batch_size,:] = hFeats
            labelMat[cntr:cntr+batch_size] = labels
        
            cntr+=batch_size
        
            if cntr>=ndata:
                break
    reconstruction_error.append(torch.mean(abs(iMat-rMat)))
# plt.plot( prod, np.asarray(reconstruction_error),label = 'SGD without momentum')
# plt.plot( prod, np.asarray(reconstruction_error),label = 'SGD with momentum')
# plt.plot( prod, np.asarray(reconstruction_error),label = 'ADAM')
plt.plot( prod, np.asarray(reconstruction_error),label = 'RMS prop')


plt.legend()
plt.ylabel("Reconstruction error")
plt.xlabel("no.of neurons")
plt.show()



