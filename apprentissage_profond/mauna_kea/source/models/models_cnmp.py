# -*- coding: utf-8 -*-
"""
Models from article https://www.hindawi.com/journals/cin/2018/2061516/
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CodingNetwork(nn.Module):
    def __init__(self, n_classes):
        super(BenchMark, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=0)    # (32, 130, 130)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=0)   # (32, 120, 120)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 114, 114)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)     # (16, 57, 57)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (16, 28, 28)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)    # (32, 14, 14)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (32, 7, 7)
        self.drop3 = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(1568, n_classes)

    def forward(self, x):
        # input is (bs, 1, 140, 140)
        x = self.pool1(self.drop1(F.relu(self.conv1(x))))
        x = self.pool2(self.drop2(F.relu(self.conv2(x))))
        x = self.pool3(self.drop3(F.relu(self.conv3(x))))
    
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        return x

