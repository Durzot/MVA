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

#class BenchMark(nn.Module):
#    def __init__(self, n_classes):
#        super(BenchMark, self).__init__()
#        self.n_classes = n_classes
#        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)      # (3, 228, 228)
#        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                       # (3, 114, 114)
#        self.drop1 = nn.Dropout2d(p=0.2)
#        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)     # (16, 57, 57)
#        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                       # (16, 28, 28)
#        self.drop2 = nn.Dropout2d(p=0.2)
#        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)    # (32, 14, 14)
#        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                       # (32, 7, 7)
#        self.drop3 = nn.Dropout2d(p=0.2)
#        
#        self.fc1 = nn.Linear(1568, n_classes)
#        self.softmax = nn.Softmax()
#
#    def forward(self, x):
#        # input is (bs, 1, 456, 456)
#        x = self.pool1(self.drop1(F.relu(self.conv1(x))))
#        x = self.pool2(self.drop2(F.relu(self.conv2(x))))
#        x = self.pool3(self.drop3(F.relu(self.conv3(x))))
#    
#        x = x.view(x.size(0), -1)
#        x= self.softmax(self.fc1(x))
#        return x

class MaunaNet2(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet3, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)      # (64, 110, 110)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                        # (64, 55, 55)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)    # (256, 28, 28)
        self.bn1 = nn.BatchNorm2d(256)
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

class MaunaNet3(nn.Module):
    def __init__(self, n_classes):
        super(MaunaNet3, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=2, padding=1)       # (3, 197, 197)
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                        # (3, 98, 98)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)      # (16, 48, 48)
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                        # (16, 24, 24)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)     # (32, 12, 12)
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                        # (32, 6, 6)

        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 392),
            nn.ReLU(inplace=True),
            nn.Linear(392, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input is (bs, 1, 395, 395)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
    
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input is (bs, 1, 224, 224)
        x = self.pool1(self.drop1(F.relu(self.conv1(x))))
        x = self.pool2(self.drop2(F.relu(self.conv2(x))))
        x = self.pool3(self.drop3(F.relu(self.conv3(x))))
    
        x = x.view(x.size(0), -1)
        return x

class BenchMarkBn(nn.Module):
    def __init__(self, n_classes):
        super(BenchMarkBn, self).__init__()
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

#class AlexNet(nn.Module):
#    def __init__(self, n_classes=4):
#        super(AlexNet, self).__init__()
#        self.features = nn.Sequential(
#            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2),
#            nn.Conv2d(64, 192, kernel_size=5, padding=2),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2),
#            nn.Conv2d(192, 384, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(384, 256, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(256, 256, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2),
#        )
#        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#        self.classifier = nn.Sequential(
#            nn.Dropout(),
#            nn.Linear(256 * 6 * 6, 4096),
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#            nn.Linear(4096, 4096),
#            nn.ReLU(inplace=True),
#            nn.Linear(4096, n_classes),
#        )
#
#    def forward(self, x):
#        x = self.features(x)
#        x = self.avgpool(x)
#        x = x.view(x.size(0), 256 * 6 * 6)
#        x = self.classifier(x)
#        return x
