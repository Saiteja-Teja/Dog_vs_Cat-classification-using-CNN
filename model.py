import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms , datasets
from google.colab import drive
import torch.nn.functional as F


class Custom_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.features=nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(32,64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
       
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(64,128,kernel_size=3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
      
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*16*16,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128,32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32,2)
    )
  def forward(self,x):
      x=self.features(x)
      x=self.classifier(x)
      return x
