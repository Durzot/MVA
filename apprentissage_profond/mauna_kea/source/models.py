# -*- coding: utf-8 -*-
"""
Models for classifying images
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaunaNet4(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet4, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 228, 228)
        self.bn1 = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 114, 114)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)     # (64, 57, 57)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (64, 29, 29)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)   # (128, 15, 15)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (128, 8, 8)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=3, stride=2, padding=1) # (1024, 4, 4)
        self.bn4 = nn.BatchNorm2d(1024)
        self.pool4 = nn.MaxPool2d(kernel_size=2)                                                       # (1024, 2, 2)
        
        self.avgpool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(1024, n_classes)

    def forward(self, x):
        # input is (bs, 1, 456, 456)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        return x
