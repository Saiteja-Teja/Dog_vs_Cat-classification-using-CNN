import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms , datasets
from google.colab import drive
import torch.nn.functional as F

custom_transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
]
                                   )
train_data=datasets.ImageFolder(root=train_dir,transform=custom_transform)
test_data=datasets.ImageFolder(root=test_dir, transform=custom_transform)
train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(test_data,batch_size=32,shuffle=False)
learning_rate=0.001
epochs=25
model=Custom_model().to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-3)
for epoch in range(epochs):
  total=0
  for batch_features,batch_labels in train_loader:
    batch_features=batch_features.to(device)
    batch_labels=batch_labels.to(device)
    # One-hot encode the labels and convert to float to match model output and loss function requirements
    one_hot_labels = F.one_hot(batch_labels, num_classes=2).float() # Assuming 2 classes based on model output
    output=model(batch_features)
    loss=criterion(output,one_hot_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total=total+loss.item()
  average=total/len(train_loader)
  print(f' Epoch {epoch+1} Loss {average}')
  
