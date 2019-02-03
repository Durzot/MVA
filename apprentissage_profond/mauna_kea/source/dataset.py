# -*- coding: utf-8 -*-
"""
Class to load data in a pytorch dataset
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image

class MaunaKea(data.Dataset):
    def __init__(self, root_img="./data/TrainingSetImagesDir", label_file="./data/TrainingSet_20aimVO.csv", 
                 split=0.8, train=True):
        self.root_img = root_img
        self.label_img = pd.read_csv(label_file)
        self.split = split
        self.train = train
        self._data = []

        self.transforms = transforms.Compose(
            [transforms.CenterCrop(456),
             transforms.ToTensor()]
        )
        
        for _, row in self.label_img.iterrows():
            path = os.path.join(root_img, row['image_filename'])
            label = row['class_number']
            self._data.append((path, label))
        
        self._limit = int(self.split*len(self._data))
        if self.train:
            self._data = self._data[:self._limit]
        else:
            self._data = self._data[self._limit:]
        
    def __getitem__(self, index):
        path, label = self._data[index]
        img = self.transforms(Image.open(path).convert('L'))
        return img, label

    def __len__(self):
        return len(self._data)

class MaunaKeaTest(data.Dataset):
    def __init__(self, root_img="./data/TestSetImagesDir/part1"):
        self.root_img = root_img
        self._data = []

        self.transforms = transforms.Compose(
            [transforms.CenterCrop(456),
             transforms.ToTensor()]
        )
        
        self._fn_img = os.listdir(self.root_img)
        self._fn_img = [fn for fn in self.fn_img if '.png' in fn]
        for fn in self._fn_img:
            path = os.path.join(root_img, fn_img)
            self._data.append(path)
        
    def __getitem__(self, index):
        path = self._data[index]
        img = self.transforms(Image.open(path).convert('L'))
        return img

    def __len__(self):
        return len(self._data)
        

