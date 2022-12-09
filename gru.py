import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split  






device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data = data[['userID','assessmentItemID','testId','Timestamp','KnowledgeTag','answerCode']]
test = test[['userID','assessmentItemID','testId','Timestamp','KnowledgeTag','answerCode']]
data = data.drop(['testId','Timestamp','assessmentItemID'],axis=1)
test = test.drop(['testId','Timestamp','assessmentItemID'],axis=1)
# data = data.astype({'userID':'float','KnowledgeTag':'float','answerCode':'float'})
print(data.dtypes)

# data = pd.get_dummies(data)

X=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

X_train_tensors = Variable(torch.Tensor(X_train.values)).to(device)
X_test_tensors = Variable(torch.Tensor(X_test.values)).to(device)

y_train_tensors = Variable(torch.Tensor(y_train.values)).to(device)
y_test_tensors = Variable(torch.Tensor(y_test.values)).to(device)

X_train_tensors_f = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])).to(device)
X_test_tensors_f = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])).to(device)

print("Training Shape", X_train_tensors_f.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_f.shape, y_test_tensors.shape)

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True).to(device)
        self.fc_1 =  nn.Linear(hidden_size, 128).to(device)
        self.fc = nn.Linear(128, num_classes).to(device)
        self.relu = nn.ReLU().to(device)
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size).to(device) 
        out = self.relu(hn).to(device)
        out = self.fc_1(out).to(device)
        out = self.relu(out).to(device)
        out = self.fc(out).to(device)
        return out

num_epochs = 200
learning_rate = 0.1 

input_size = 2 
hidden_size = 64
num_layers = 1 

num_classes = 1 
model = GRU(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1]).to(device)

criterion = torch.nn.MSELoss()    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):
    outputs = model.forward(X_train_tensors_f).to(device)
    optimizer.zero_grad()  
    loss = criterion(outputs, y_train_tensors)
    loss.backward() 
 
    optimizer.step() 
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 


# TX_train_tensors = Variable(torch.Tensor(X_train.values))
# TX_test_tensors = Variable(torch.Tensor(X_test.values))

# Ty_train_tensors = Variable(torch.Tensor(y_train.values))
# Ty_test_tensors = Variable(torch.Tensor(y_test.values))

# X_train_tensors_f = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
# X_test_tensors_f = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

