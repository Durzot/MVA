# -*- coding: utf-8 -*-
"""
Models for classifying images
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

#########################
# Simple architectures
#########################

class MaunaNet2(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet2, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)      # (64, 110, 110)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                        # (64, 55, 55)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)    # (256, 28, 28)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                        # (256, 14, 14)
        
        self.avgpool = nn.AvgPool2d(14)                                                                 # (256, 1, 1)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # input is (bs, 3, 224, 224)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        # input is (bs, 3, 224, 224)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class MaunaNet3(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet3, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)      # (64, 110, 110)
        self.bn1 = nn.BatchNorm2d(64)                                                                   
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                        # (64, 55, 55)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)    # (256, 28, 28)
        self.bn2 = nn.BatchNorm2d(256)                                                                   
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                        # (256, 14, 14)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)   # (512, 7, 7)
        self.bn3 = nn.BatchNorm2d(512)                                                                   
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                        # (512, 4, 4)

        self.avgpool = nn.AvgPool2d(4)                                                                  # (512, 1, 1)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # input is (bs, 3, 224, 224)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        # input is (bs, 3, 224, 224)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class MaunaNet4(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet4, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)      # (64, 110, 110)
        self.bn1 = nn.BatchNorm2d(64)                                                                   
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                        # (64, 55, 55)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)    # (256, 28, 28)
        self.bn2 = nn.BatchNorm2d(256)                                                                   
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)   # (512, 14, 14)
        self.bn3 = nn.BatchNorm2d(512)                                                                   
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)  # (1024, 7, 7)
        self.bn4 = nn.BatchNorm2d(1024)                                                                   

        self.avgpool = nn.AvgPool2d(7)                                                                  # (1024, 1, 1)
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        # input is (bs, 3, 224, 224)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        # input is (bs, 3, 224, 224)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class BenchMark(nn.Module):
    def __init__(self, n_classes):
        super(BenchMark, self).__init__()
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

    def forward(self, x):
        # input is (bs, 1, 224, 224)
        x = self.pool1(self.drop1(F.relu(self.conv1(x))))
        x = self.pool2(self.drop2(F.relu(self.conv2(x))))
        x = self.pool3(self.drop3(F.relu(self.conv3(x))))
    
        x = x.view(x.size(0), -1)
        return x

#########################
# Classical architectures
#########################

class AlexNet(nn.Module):
    def __init__(self, n_classes, fz_depth):
        super(AlexNet,self).__init__()
        network = torchvision.models.alexnet(pretrained=True)

        # Freeze first fz_depth parameters
        fz_count = 0
        for nparam, param in network.named_parameters():
            if fz_count < fz_depth:
                print("Freezing parameter %s" % nparam)
                param.requires_grad = False
                fz_count += 1
            else:
                break

        # Feature layers
        child = list(network.children())[0]
        self.conv1 = nn.Sequential(*list(child)[0:3])
        self.conv2 = nn.Sequential(*list(child)[3:6]) 
        self.conv3 = nn.Sequential(*list(child)[6:8])
        self.conv4 = nn.Sequential(*list(child)[8:10])
        self.conv5 = nn.Sequential(*list(child)[10:13])
    	    
        # Classifying layers
        child = list(network.children())[1]
        self.fc1 = nn.Sequential(*list(child)[0:3])
        self.fc2 = nn.Sequential(*list(child)[3:6])
        self.fc3 = nn.Linear(4096, n_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def get_features(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, n_classes, fz_depth):
        super(ResNet18,self).__init__()
        network = torchvision.models.resnet18(pretrained=True)

        # Freeze first fz_depth parameters
        fz_count = 0
        for nparam, param in network.named_parameters():
            if fz_count < fz_depth:
                print("Freezing parameter %s" % nparam)
                param.requires_grad = False
                fz_count += 1
            else:
                break

        # First convolutional layer
        self.conv1 = nn.Sequential(*list(network.children())[0:4])
        self.basicblock1 = list(network.children())[4]
        self.basicblock2 = list(network.children())[5]
        self.basicblock3 = list(network.children())[6]
        self.basicblock4 = list(network.children())[7]
        self.avgpool = list(network.children())[8]
        
        # Classifying layers
        self.fc1 = nn.Linear(512, n_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.basicblock1(x)
        x = self.basicblock2(x)
        x = self.basicblock3(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def get_features(self,x):
        x = self.conv1(x)
        x = self.basicblock1(x)
        x = self.basicblock2(x)
        x = self.basicblock3(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        return x
