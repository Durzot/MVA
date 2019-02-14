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

import numbers
import random
from sklearn.model_selection import train_test_split

radius = 278

class RandomCropCircle(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        radius (int): Radius of the circular image from which to extract a 
            rectangular crop.   
    """

    def __init__(self, size, radius):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.radius = radius

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        h, w = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        # Some images have black bands on the top and the bottom
        # We quantify the height of that band in a margin from which we won't sample
        pixels = np.array(img).astype("float32")
        margin = next((i for i,pixel in enumerate(pixels[:, 260]) if pixel !=0), None)  
    
        xmin = max(int(w/2-np.sqrt(self.radius**2 - th**2/4.)+1), 0)
        xmax = int(w/2+np.sqrt(self.radius**2 - th**2/4.)+1) - tw
        x = random.randint(xmin, xmax)
        
        ymax = int(h/2 - th + min(np.sqrt(self.radius**2 - (x-w/2)**2), np.sqrt(self.radius**2 - (x+tw-w/2)**2)))
        ymin = int(h/2 - min(np.sqrt(self.radius**2 - (x-w/2)**2), np.sqrt(self.radius**2 - (x+tw-w/2)**2)))
        y = random.randint(max(margin, ymin), ymax)

        return img.crop((x, y, x + tw, y + th))

class MaunaKea(data.Dataset):
    def __init__(self, root_img="./data/TrainingSetImagesDir", label_file="./data/TrainingSet_20aimVO.csv", 
                 train_size=0.8, train=True, data_aug=0, random_state=0):
        self.root_img = root_img
        self.label_img = pd.read_csv(label_file)
        self.train_size = train_size
        self.train = train
        self.data_aug = data_aug
        self.random_state = random_state
        self._data = []

        if self.data_aug:
            self.transforms = transforms.Compose(
                [RandomCropCircle(224, radius),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose(
                [transforms.CenterCrop(456),
                 transforms.ToTensor()])
       
        train_idx, test_idx = train_test_split(self.label_img.index, 
                                               train_size=self.train_size,
                                               stratify=self.label_img.class_number,
                                               random_state=self.random_state)

        for idx, row in self.label_img.iterrows():
            if self.train and idx in train_idx:
                path = os.path.join(root_img, row['image_filename'])
                label = row['class_number']
                self._data.append((path, label))
            elif not self.train and idx in test_idx:
                path = os.path.join(root_img, row['image_filename'])
                label = row['class_number']
                self._data.append((path, label))
                
    def __getitem__(self, index):
        path, label = self._data[index]
        img = self.transforms(Image.open(path).convert('L'))
        return img, label

    def __len__(self):
        return len(self._data)

class MaunaKeaTest(data.Dataset):
    def __init__(self, root_img="./data/TestSetImagesDir/part1", data_aug=0):
        self.root_img = root_img
        self.data_aug = data_aug
        self._data = []

        if self.data_aug:
            self.transforms = transforms.Compose(
                [RandomCropCircle(224, radius),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose(
                [transforms.CenterCrop(456),
                 transforms.ToTensor()])
        
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
        

