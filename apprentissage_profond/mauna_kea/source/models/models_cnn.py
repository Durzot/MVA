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

class BenchMark(nn.Module):
    def __init__(self, n_classes):
        super(BenchMark, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 228, 228)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 114, 114)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)     # (16, 57, 57)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (16, 28, 28)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)    # (32, 14, 14)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (32, 7, 7)
        self.drop3 = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(1568, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 456, 456)
        x = self.pool1(self.drop1(F.relu(self.conv1(x))))
        x = self.pool2(self.drop2(F.relu(self.conv2(x))))
        x = self.pool3(self.drop3(F.relu(self.conv3(x))))
    
        x = x.view(x.size(0), -1)
        x= self.softmax(self.fc1(x))
        return x

class BenchMarkAug(nn.Module):
    def __init__(self, n_classes):
        super(BenchMarkAug, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 112, 112)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 56, 56)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)     # (16, 28, 28)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (16, 14, 14)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1)    # (64, 7, 7)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (64, 3, 3)
        self.drop3 = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(576, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 224, 224)
        x = self.pool1(self.drop1(F.relu(self.conv1(x))))
        x = self.pool2(self.drop2(F.relu(self.conv2(x))))
        x = self.pool3(self.drop3(F.relu(self.conv3(x))))
    
        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc1(x))
        return x

class BenchMarkAugBn(nn.Module):
    def __init__(self, n_classes):
        super(BenchMarkAugBn, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 112, 112)
        self.bn1 = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 56, 56)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)     # (16, 28, 28)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (16, 14, 14)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1)    # (64, 7, 7)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (64, 3, 3)
        self.drop3 = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(576, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 224, 224)
        x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))
    
        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc1(x))
        return x

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
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # (256, 4, 4)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)                                                       # (256, 2, 2)
        
        self.avgpool = nn.AvgPool2d(2)                                                                 # (256, 1, 1)
        self.fc1 = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 456, 456)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc1(x))
        return x

class MaunaNet4Drop(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet4Drop, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 228, 228)
        self.bn1 = nn.BatchNorm2d(3)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 114, 114)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)     # (64, 57, 57)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (64, 29, 29)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)   # (128, 15, 15)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (128, 8, 8)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # (256, 4, 4)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(p=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)                                                       # (256, 2, 2)
        
        self.avgpool = nn.AvgPool2d(2)                                                                 # (256, 1, 1)
        self.fc1 = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 456, 456)
        x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.drop4(F.relu(self.bn4(self.conv4(x)))))
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc1(x))
        return x

class MaunaNet4Aug(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet4Aug, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 112, 112)
        self.bn1 = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)     # (64, 56, 56)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (64, 28, 28)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)   # (128, 28, 28)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (128, 14, 14)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 14, 14)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)                                                       # (256, 7, 7)
        
        self.avgpool = nn.AvgPool2d(7)                                                                 # (256, 1, 1)
        self.fc1 = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 224, 224)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc1(x))
        return x

class MaunaNet4AugDrop(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet4AugDrop, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 112, 112)
        self.bn1 = nn.BatchNorm2d(3)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)     # (64, 56, 56)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (64, 28, 28)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)   # (128, 28, 28)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (128, 14, 14)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 14, 14)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(p=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)                                                       # (256, 7, 7)
        
        self.avgpool = nn.AvgPool2d(7)                                                                 # (256, 1, 1)
        self.fc1 = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 224, 224)
        x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.drop4(F.relu(self.bn4(self.conv4(x)))))
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc1(x))
        return x
