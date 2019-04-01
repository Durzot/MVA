# -*- coding: utf-8 -*-
"""
Class to load data in a pytorch dataset
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""
import os
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image

import numbers
import random
from sklearn.model_selection import train_test_split

from skimage.feature import greycomatrix, greycoprops

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
        if len(pixels.shape) == 3:
            # Input images are in grayscale anyway
            pixels = pixels[:, :, 0]
        margin = next((i for i,pixel in enumerate(pixels[:, 260]) if pixel !=0), None)  
    
        xmin = max(int(w/2-np.sqrt(self.radius**2 - th**2/4.)+1), 0)
        xmax = int(w/2+np.sqrt(self.radius**2 - th**2/4.)+1) - tw
        x = random.randint(xmin, xmax)
        
        ymax = int(h/2 - th + min(np.sqrt(self.radius**2 - (x-w/2)**2), np.sqrt(self.radius**2 - (x+tw-w/2)**2)))
        ymin = int(h/2 - min(np.sqrt(self.radius**2 - (x-w/2)**2), np.sqrt(self.radius**2 - (x+tw-w/2)**2)))
        y = random.randint(max(margin, ymin), ymax)

        return img.crop((x, y, x + tw, y + th))

#class MaunaKea(data.Dataset):
#    def __init__(self, root_img="./data/TrainingSetImagesDir", label_file="./data/TrainingSet_20aimVO.csv", 
#                 test_size=0.2, train=True, data_aug=0, random_state=0):
#        self.root_img = root_img
#        self.label_img = pd.read_csv(label_file)
#        self.test_size = test_size
#        self.train = train
#        self.data_aug = data_aug
#        self.random_state = random_state
#        self._data = []
#
#        if self.data_aug:
#            self.transforms = transforms.Compose(
#                [RandomCropCircle(224, radius),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor()])
#        else:
#            self.transforms = transforms.Compose(
#                [transforms.CenterCrop(456),
#                 transforms.ToTensor()])
#
##        Wrong way to split
##        train_idx, test_idx = train_test_split(self.label_img.index, 
##                                               test_size=self.test_size,
##                                               stratify=self.label_img.class_number,
##                                               random_state=self.random_state)
#
#        patient_col = self.label_img.image_filename.apply(lambda x: x.split("_")[-1].split(".")[0]).astype(int)
#        self.label_img.insert(0, column='patient', value=patient_col)     
#
#        self._train_pat, self._test_pat = train_test_split(self.label_img.patient.unique(), 
#                                                           test_size=self.test_size,
#                                                           random_state=self.random_state)
#
#        for _, row in self.label_img.iterrows():
#            if self.train and row['patient'] in self._train_pat:
#                path = os.path.join(root_img, row['image_filename'])
#                label = row['class_number']
#                self._data.append((path, label))
#            elif not self.train and row['patient'] in self._test_pat:
#                path = os.path.join(root_img, row['image_filename'])
#                label = row['class_number']
#                self._data.append((path, label))
#                
#    def __getitem__(self, index):
#        path, label = self._data[index]
#        img = self.transforms(Image.open(path).convert('L'))
#        return img, label
#
#    def __len__(self):
#        return len(self._data)
#
#class MaunaKeaTest(data.Dataset):
#    def __init__(self, root_img="./data/TestSetImagesDir/part_1", data_aug=0):
#        self.root_img = root_img
#        self.data_aug = data_aug
#        self._data = []
#
#        if self.data_aug:
#            self.transforms = transforms.Compose(
#                [RandomCropCircle(224, radius),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor()])
#        else:
#            self.transforms = transforms.Compose(
#                [transforms.CenterCrop(456),
#                 transforms.ToTensor()])
#        
#        self._fn_img = os.listdir(self.root_img)
#        self._fn_img = [fn for fn in self._fn_img if '.png' in fn]
#        for fn in self._fn_img:
#            path = os.path.join(root_img, fn)
#            self._data.append((fn, path))
#        
#    def __getitem__(self, index):
#        fn, path = self._data[index]
#        img = self.transforms(Image.open(path).convert('L'))
#        return fn, img
#
#    def __len__(self):
#        return len(self._data)

class MaunaKea(data.Dataset):
    def __init__(self, root_img="./data/TrainingSetImagesDir", label_file="./data/TrainingSet_20aimVO.csv", 
                 test_size=0.2, train=True, data_aug=0, crop_size=224, rgb=1, img_size=224, random_state=0):
        self.root_img = root_img
        self.label_img = pd.read_csv(label_file)
        self.test_size = test_size
        self.train = train
        self.data_aug = data_aug
        self.rgb = rgb
        self.random_state = random_state
        self._data = []

        if self.data_aug:
            self.transforms = transforms.Compose(
                [RandomCropCircle(crop_size, radius),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.Resize(img_size, interpolation=2),
                 transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose(
                [transforms.CenterCrop(395),
                 transforms.Resize(img_size, interpolation=2),
                 transforms.ToTensor()])

        patient_col = self.label_img.image_filename.apply(lambda x: x.split("_")[-1].split(".")[0]).astype(int)
        self.label_img.insert(0, column='patient', value=patient_col)     

        self._train_pat, self._test_pat = train_test_split(self.label_img.patient.unique(), 
                                                           test_size=self.test_size,
                                                           random_state=self.random_state)

        for _, row in self.label_img.iterrows():
            if self.train and row['patient'] in self._train_pat:
                fn = row['image_filename']
                label = row['class_number']
                path = os.path.join(root_img, fn)
                self._data.append((fn, label, path))
            elif not self.train and row['patient'] in self._test_pat:
                fn = row['image_filename']
                label = row['class_number']
                path = os.path.join(root_img, fn)
                self._data.append((fn, label, path))
                
    def __getitem__(self, index):
        fn, label, path = self._data[index]
        if self.rgb:
            img = self.transforms(Image.open(path))
        else:
            img = self.transforms(Image.open(path).convert('L'))
        return fn, label, img

    def __len__(self):
        return len(self._data)

class MaunaKeaTest(data.Dataset):
    def __init__(self, root_img="./data/TestSetImagesDir/part_1", data_aug=0, crop_size=224, rgb=1, img_size=224):
        self.root_img = root_img
        self.data_aug = data_aug
        self.rgb = rgb
        self._data = []

        if self.data_aug:
            self.transforms = transforms.Compose(
                [RandomCropCircle(crop_size, radius),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.Resize(img_size, interpolation=2),
                 transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose(
                [transforms.CenterCrop(395),
                 transforms.Resize(img_size, interpolation=2),
                 transforms.ToTensor()])
        
        self._fn_img = os.listdir(self.root_img)
        self._fn_img = [fn for fn in self._fn_img if '.png' in fn]
        for fn in self._fn_img:
            path = os.path.join(root_img, fn)
            self._data.append((fn, path))
        
    def __getitem__(self, index):
        fn, path = self._data[index]
        if self.rgb:
            img = self.transforms(Image.open(path))
        else:
            img = self.transforms(Image.open(path).convert('L'))
        return fn, img

    def __len__(self):
        return len(self._data)

#####################################################
# Extract textural features to feed into a classifier
#####################################################

# See Robert M.Haralick,K. Shanmugam,It‟shak Distein ,
# “Textural Features for Image Classification” ,IEEE
# Transactions on systems, Man and Cybernetics, Vol 1973

class MaunaTexturalFeatures(object):
    def __init__(self, root_img="./data/TrainingSetImagesDir", label_file="./data/TrainingSet_20aimVO.csv", 
                 feature_mean=True, feature_range=True, feature_std=True):
        self.root_img = root_img
        self.label_img = pd.read_csv(label_file)
        self.feature_mean = feature_mean
        self.feature_range = feature_range
        self.feature_std = feature_std

        # We extract the biggest square from the image that excludes black edges
        # If you know the radius R, the width of this square is R*sqrt(2)
        # As R is 278, this is approx. 395
        self.transforms = transforms.Compose(
            [transforms.CenterCrop(395)])

        patient_col = self.label_img.image_filename.apply(lambda x: x.split("_")[-1].split(".")[0]).astype(int)
        self.label_img.insert(0, column='patient', value=patient_col)     

    def get_data(self):
        data = pd.DataFrame()

        for _, row in tqdm(self.label_img.iterrows()):
           data_row = {}
           data_row['patient'] = row['patient']
           data_row['label'] = row['class_number']

           path = os.path.join(self.root_img, row['image_filename'])
           img = np.array(self.transforms(Image.open(path).convert('L')))
           features = self._get_features(img)

           data_row = dict(data_row, **features)
           data = data.append(data_row, ignore_index=True)

        return data

    def _get_features(self, img):
        distances = [1]
        angles = [0, np.pi*35/180, np.pi*90/180, np.pi*135/180]
        P = greycomatrix(img, distances, angles)

        f = {}
        (num_level, num_level, num_dist, num_angle) = P.shape

        # normalize each GLCM
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums
        
        #######################
        # Angular second moment
        #######################
        results = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]  
        f['ams'] = results

        #######################
        # Contrast
        #######################
        I, J = np.ogrid[0:num_level, 0:num_level]
        weights = (I - J) ** 2
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]
        f['contrast'] = results

        #######################
        # Correlation
        #######################
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        std_i = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_i) ** 2),
                                           axes=(0, 1))[0, 0])
        std_j = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_j) ** 2),
                                           axes=(0, 1))[0, 0])
        cov = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
                                 axes=(0, 1))[0, 0]

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = mask_0 == False
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
        f['correlation'] = results

        #######################
        # Sum of squares
        #######################
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1))
        I = np.repeat(I, num_level, axis=1).reshape((num_level, num_level, 1, 1))
        diff_i = np.square(I - np.apply_over_axes(np.sum, P, axes=(0, 1))[0, 0])
        results = np.apply_over_axes(np.sum, (diff_i * P), axes=(0, 1))[0, 0]
        f['sum_sq'] = results

        #######################
        # Inverse difference moment
        #######################
        I, J = np.ogrid[0:num_level, 0:num_level]
        weights = 1. / (1. + (I - J) ** 2)
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]
        f['inv_diff_mom'] = results

        P_ij = np.zeros(((2*num_level-1), num_dist, num_angle))
        P_diff_ij = np.zeros((num_level, num_dist, num_angle))
        for a in range(num_angle):
            for d in range(num_dist):
                for k in range(num_level):
                    for i in range(max(k-num_level+1, 0), min(k, num_level)):
                        P_ij[k, d, a] += P[i, k-i, d, a]
                    for i in range(num_level):
                        if i-k >= 0:
                            P_diff_ij[k, d, a] += P[i, i-k, d, a]
                        if i+k < num_level:
                            P_diff_ij[k, d, a] += P[i, i+k, d, a]
                for k in range(num_level, 2*num_level-1):
                    for i in range(max(k-num_level+1, 0), min(k, num_level)):
                        P_ij[k, d, a] += P[i, k-i, d, a]

        
        #######################
        # Sum average
        #######################
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(2*num_level-1)).reshape((2*num_level-1, 1, 1))
        results = np.apply_over_axes(np.sum, (I*P_ij), axes=(0))[0]
        f['sum_avg'] = results

        #######################
        # Sum variance
        #######################
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.square(np.array(range(2*num_level-1)).reshape((2*num_level-1, 1, 1)) - f['sum_avg'])
        results = np.apply_over_axes(np.sum, (I*P_ij), axes=(0))[0]
        f['sum_var'] = results

        #######################
        # Sum entropy
        #######################
        eps = 1e-15
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        log_P_ij = np.log(P_ij + eps)
        results = np.apply_over_axes(np.sum, -(log_P_ij*P_ij), axes=(0))[0]
        f['sum_ent'] = results

        #######################
        # Entropy
        #######################
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        results = np.apply_over_axes(np.sum, -(np.log(P+eps)*P), axes=(0, 1))[0, 0]
        f['ent'] = results
        
        #######################
        # Difference Variance
        #######################
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1))
        diff_avg = np.apply_over_axes(np.sum, (I*P_diff_ij), axes=(0))[0]

        I = np.square(np.array(range(num_level)).reshape((num_level, 1, 1)) - diff_avg)
        results = np.apply_over_axes(np.sum, (I*P_diff_ij), axes=(0))[0]
        f['diff_var'] = results

        #######################
        # Diff entropy
        #######################
        eps = 1e-15
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        log_P_diff_ij = np.log(P_diff_ij + eps)
        results = np.apply_over_axes(np.sum, -(log_P_diff_ij*P_diff_ij), axes=(0))[0]
        f['diff_ent'] = results

        P_i = np.apply_over_axes(np.sum, P, axes=(1))
        P_j = np.apply_over_axes(np.sum, P, axes=(0))
        P_iP_j = np.zeros((num_level, num_level, num_dist, num_angle))

        for a in range(num_angle):
            for d in range(num_dist):
                P_iP_j[:, :, d, a] = np.dot(P_i[:, :, d, a], P_j[:, :, d, a])
                
        HX = np.apply_over_axes(np.sum, -(np.log(P_i+eps)*P_i), axes=(0,1))[0, 0]
        HY = np.apply_over_axes(np.sum, -(np.log(P_j+eps)*P_j), axes=(0,1))[0, 0]
        HXY = f['ent']

        HXY1 = np.apply_over_axes(np.sum, -(np.log(P_iP_j+eps)*P), axes=(0, 1))[0, 0]
        HXY2 = np.apply_over_axes(np.sum, -(np.log(P_iP_j+eps)*P_iP_j), axes=(0, 1))[0, 0]

        f['m1_corr'] = (HXY - HXY1)/np.fmax(HX, HY)
        f['m2_corr'] = np.sqrt(1-np.exp(-2.*(HXY2-HXY)))

        features = {}
        for d in range(num_dist):
            for k in f.keys():
                if self.feature_mean:
                    features['%s_%d_mean' % (k, d)] = f[k][d, :].mean()
                if self.feature_range:
                    features['%s_%d_range' % (k, d)] = f[k][d, :].max() -  f[k][d, :].min() 
                if self.feature_std:
                    features['%s_%d_std' % (k, d)] = f[k][d, :].std()
        return features

