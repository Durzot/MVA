# -*- coding: utf-8 -*-
"""
Functions to train models on the train set
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd
import time
import datetime

import torch
sys.path.append("./source")
from auxiliary.dataset import *
from auxiliary.utils import *
from models.models_cnn import *

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M") 

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--data_aug', type=int, default=0 , help='1 for data augmentation')
parser.add_argument('--crop_size', type=int, default=224 , help='crop size if data augmentation is 1')
parser.add_argument('--img_size', type=int, default=224 , help='size of input images to model')
parser.add_argument('--rgb', type=int, default=1, help='1 for 3 channel input, 0 for 1 channel input')
parser.add_argument('--n_classes', type=int, default=4, help='number of classes')
parser.add_argument('--model_type', type=str, default='cnn_Mauna_decay_20_20',  help='type of model')
parser.add_argument('--model_name', type=str, default='MaunaNet2',  help='name of the model for log')
parser.add_argument('--optimizer', type=str, default="sgd",  help='optimizer used to train the model')
parser.add_argument('--cuda', type=int, default=0, help='set to 1 to use cuda')
parser.add_argument('--data_textural', type=str, default='./data/TransformedData/data_textural_train.csv', help='path data')
parser.add_argument('--n_splits', type=int, default=5, help='number of splits for cv of params')
parser.add_argument('--random_state', type=int, default=50, help='random state for the split of data')
opt = parser.parse_args()

model_path = 'trained_models/%s/%s_%s.pth' % (opt.model_type, opt.model_name, opt.optimizer)

if not os.path.exists(model_path):
    raise ValueError("There is no model %s" % model_path)

# ========================== HIGH LEVEL FEATURES ========================== #
dataset_train = MaunaKea(train=True, data_aug=opt.data_aug, crop_size=opt.crop_size, random_state=opt.random_state, rgb=opt.rgb, img_size=opt.img_size)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

dataset_test = MaunaKea(train=False, data_aug=opt.data_aug, crop_size=opt.crop_size, random_state=opt.random_state, rgb=opt.rgb, img_size=opt.img_size)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

print('training set size %d' % len(dataset_train))
print('test set size %d' % len(dataset_test))

# Load network
network = eval("%s(n_classes=opt.n_classes)" % opt.model_name)
if opt.cuda:
    network.load_state_dict(torch.load(model_path))  
    network.cuda()
else:
    network.load_state_dict(torch.load(model_path, map_location='cpu'))
network = network.eval()
print("Weights from %s loaded" % model_path)

# Get high-level features from train set
list_features = []
list_label = []
list_fn = []
for batch, (fn, label, data) in enumerate(loader_train):
    if opt.cuda:
        data = data.cuda()
    list_features.append(network.get_features(data).cpu().data.numpy())
    list_label.append(label.cpu().data.numpy())
    list_fn.append(list(fn))

    if batch > 2:
        break

data_hl_feat = np.concatenate((np.concatenate(list_fn, axis=0).reshape(-1,1), 
                               np.concatenate(list_label, axis=0).reshape(-1, 1), 
                               np.concatenate(list_features, axis=0)), axis=1)

columns = ['image_filename', 'class_number'] + ['hf_%d' % (i-1) for i in range(2, data_hl_feat.shape[1])]
data_hl_feat = pd.DataFrame(data_hl_feat, columns=columns)

# ========================== TRADITIONAL FEATURES ========================== #
if opt.data_textural is not None:
    data_textural = pd.read_csv(opt.data_textural, header='infer')
else:
    dataset_textural = MaunaTexturalFeatures(feature_mean=True, feature_std=True, feature_range=True)
    data_textural =  dataset_textural.get_data()
    data_textural.to_csv('./TransformedData/data_textural_train.csv', index=False)

dataset_textural_1 = MaunaTexturalFeatures(root_img="./data/TestSetImagesDir/part_1", train=False, feature_mean=True, feature_std=True, feature_range=True)
data_textural_1 = dataset_textural_1.get_data()
data_textural_1.to_csv('./data/TransformedData/data_textural_test_1.csv', index=False)

dataset_textural_2 = MaunaTexturalFeatures(root_img="./data/TestSetImagesDir/part_2", train=False, feature_mean=True, feature_std=True, feature_range=True)
data_textural_2 = dataset_textural_2.get_data()
data_textural_2.to_csv('./data/TransformedData/data_textural_test_2.csv', index=False)



X = data_textural.loc[:, [x for x in data_textural.columns if x not in ['patient', 'image_filename', 'label']]].values
y = data_textural.loc[:, 'label'].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=opt.test_size, stratify=y,
                                                  random_state=opt.random_state)

# Normalize the data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_val_std = sc.transform(X_val)

# ============================ PLOT FEATURES DISTRIBUTION ============================ #
feature_names = [x for x in data_textural.columns if x not in ['patient', 'image_filename', 'label']]

colors = ['r','g','b','orange']

# DISTRIBUTION OF 6 FIRST FEATURES
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
ax = ax.flatten()
for i in range(6):
    fname = feature_names[i]
    for l, c in zip(np.unique(y_train), colors):
        ax[i].hist(X_train_std[y_train==l, i], bins=100, density=True, alpha=0.5, color=c, label="cat %d" % l)
        ax[i].set_title("Variable %s" % fname, fontweight="bold", fontsize=15)
        ax[i].legend(loc="best", fontsize=12)
        ax[i].set_xlim([-5, 5])
plt.suptitle("Distribution des variables de texture normalisées", fontsize=20, fontweight="bold")
plt.savefig('graphs/textural/feat_distrib_1.png')

# DISTRIBUTION OF 6 LAST FEATURES
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
ax = ax.flatten()
for i in range(6):
    fname = feature_names[i+6]
    for l, c in zip(np.unique(y_train), colors):
        ax[i].hist(X_train_std[y_train==l, i+6], bins=100, density=True, alpha=0.5, color=c, label="cat %d" % l)
        ax[i].set_title("Variable %s" % fname, fontweight="bold", fontsize=15)
        ax[i].legend(loc="best", fontsize=12)
        ax[i].set_xlim([-5, 5])
plt.suptitle("Distribution des variables de texture normalisées", fontsize=20, fontweight="bold")
plt.savefig('graphs/textural/feat_distrib_2.png')

# ============================ PCA ============================ #
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\n Eigenvalues \n%s'%eigen_vals)

tot = sum(eigen_vals)
expl_var = [i/tot for i in sorted(eigen_vals,reverse=True)]
cum_expl_var = np.cumsum(expl_var)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
ax.bar(np.arange(1, X_train_std.shape[1]+1), expl_var,alpha=0.5,align='center',
        label='var. expl. individuelle')
ax.step(np.arange(1, X_train_std.shape[1]+1), cum_expl_var, where='mid', label='var. expl. cumulée')
ax.set_xlabel('Indice composante principale', fontweight="bold", fontsize=20)
ax.set_ylabel('Ratio de variance expliquée',  fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=15)
plt.savefig('graphs/textural/pca_explained_var.png')

# PROJECT
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_std)
X_val_pca = pca.transform(X_val_std)

# PLOT PROJECTION ON PC 1 AND 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
colors = ['r','g','b','orange']
markers = ['+','x','o', '^']
for l, c, m in zip(np.unique(y_train), colors,markers):
    ax.scatter(X_train_pca[y_train==l,0],
               X_train_pca[y_train==l,1],
               color=c,marker=m, label="cat %d" % l)

ax.set_xlabel('PC 1', fontweight="bold", fontsize=20)
ax.set_ylabel('PC 2', fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=20)
#ax.set_xlim([-10, 10])
#ax.set_ylim([-10, 10])
ax.set_title("Projection des variables de texture sur PC 1 et 2", fontweight="bold", fontsize=25)
plt.savefig('graphs/textural/pca_12.png')

# PLOT PROJECTION ON PC 3 AND 4
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
colors = ['r','g','b','orange']
markers = ['+','x','o', '^']
for l, c, m in zip(np.unique(y_train), colors,markers):
    ax.scatter(X_train_pca[y_train==l,2],
               X_train_pca[y_train==l,3],
               color=c,marker=m,label="cat %d" % l)

ax.set_xlabel('PC 3', fontweight="bold", fontsize=20)
ax.set_ylabel('PC 4', fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=20)
ax.set_title("Projection des variables de texture sur PC 1 et 2", fontweight="bold", fontsize=25)
plt.savefig('graphs/textural/pca_34.png')

# ============================ Kernel PCA ============================ #

# PROJECT
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_train_kpca = kpca.fit_transform(X_train_std)
X_val_kpca = kpca.transform(X_val_std)

# PLOT PROJECTION ON PC 1 AND 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
colors = ['r','g','b','orange']
markers = ['+','x','o', '^']
for l, c, m in zip(np.unique(y_train), colors,markers):
    ax.scatter(X_train_kpca[y_train==l,0],
               X_train_kpca[y_train==l,1],
               color=c,marker=m, label="cat %d" % l)

ax.set_xlabel('KPC 1', fontweight="bold", fontsize=20)
ax.set_ylabel('KPC 2', fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=20)
#ax.set_xlim([-10, 10])
#ax.set_ylim([-10, 10])
ax.set_title("Projection des variables de texture sur KPC 1 et 2", fontweight="bold", fontsize=25)
plt.savefig('graphs/textural/kpca_12.png')

# =========================== DATA SPLIT ============================ #
# Build train/test splits at patient level
kf = KFold(n_splits=opt.n_splits, shuffle=True, random_state=0)
cv_splits = []
for train_pat, test_pat in kf.split(data_textural.patient.unique()):
    train_idx = data_textural.loc[data_textural.patient.isin(train_pat)].index
    test_idx = data_textural.loc[data_textural.patient.isin(test_pat)].index
    cv_splits.append((train_idx, test_idx))

# ======================== MODEL AND GRIDSEARCH ========================= #
# Build the pipeline and the gridsearch
pipe = Pipeline([('scaler', Scaler(mean=True, norm_ord=2)), ('clf', SVC())])
param_grid = {'clf__kernel' : ['linear', 'rbf'], 
              'clf__C' : np.logspace(-2, 2, num=5)}

gs = GridSearchCV(estimator=pipe,
                 param_grid=param_grid,
                 scoring='accuracy',
                 cv=cv_splits)

# ========================== FIT AND CURVES ========================== #
gs.fit(X, y)



