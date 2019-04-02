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
        super(CodingNetwork, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, stride=1, padding=0)    # (32, 130, 130)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=0)   # (32, 120, 120)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2)                                             # (32, 58, 58)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=0)    # (64, 50, 50)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=2)                                             # (64, 23, 23)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=0)   # (128, 16, 16)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=9, stride=1, padding=0)  # (256, 8, 8)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=8, stride=1, padding=0)  # (256, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
     
        self.fc1 = nn.Linear(256, n_classes)

    def forward(self, x):
        # input is (bs, 1, 140, 140)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def get_features(self, x):
        # input is (bs, 1, 140, 140)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        x = x.view(x.size(0), -1)
        return x
