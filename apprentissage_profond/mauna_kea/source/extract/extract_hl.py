# -*- coding: utf-8 -*-
"""
Script to extract high-level features from CNN networks
Also plot the distribution of each feature and perform some PCA analysis.
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd
import datetime
from tqdm import tqdm

import torch
sys.path.append("./source")
from auxiliary.dataset import *
from auxiliary.utils import *
from models.models_cnn import *
from models.models_cnmp import *

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

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
parser.add_argument('--random_state', type=int, default=50, help='random state for the split of data')
parser.add_argument('--plot_distrib', type=int, default=0, help='1 for plotting distribution of individual features')
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
print("\nWeights from %s loaded\n" % model_path)

# Get high-level features from train set
print("Extracting high-level features from train set ...")
list_features = []
list_label = []
list_fn = []
for batch, (fn, label, data) in tqdm(enumerate(loader_train)):
    if opt.cuda:
        data = data.cuda()
    list_features.append(network.get_features(data).cpu().data.numpy())
    list_label.append(label.cpu().data.numpy())
    list_fn.append(list(fn))
print("Done !\n")

data_hl_feat_train = np.concatenate((np.concatenate(list_fn, axis=0).reshape(-1,1), 
                                     np.concatenate(list_label, axis=0).reshape(-1, 1), 
                                     np.concatenate(list_features, axis=0)), axis=1)

columns = ['image_filename', 'class_number'] + ['hf_%d' % (i-1) for i in range(2, data_hl_feat_train.shape[1])]
data_hl_feat_train = pd.DataFrame(data_hl_feat_train, columns=columns)

# Get high-level features from (internal) test set
print("Extracting high-level features from test set ...")
list_features = []
list_label = []
list_fn = []
for batch, (fn, label, data) in tqdm(enumerate(loader_test)):
    if opt.cuda:
        data = data.cuda()
    list_features.append(network.get_features(data).cpu().data.numpy())
    list_label.append(label.cpu().data.numpy())
    list_fn.append(list(fn))
print("Done !\n")

data_hl_feat_test = np.concatenate((np.concatenate(list_fn, axis=0).reshape(-1,1), 
                                    np.concatenate(list_label, axis=0).reshape(-1, 1), 
                                    np.concatenate(list_features, axis=0)), axis=1)

columns = ['image_filename', 'class_number'] + ['hf_%d' % (i-1) for i in range(2, data_hl_feat_test.shape[1])]
data_hl_feat_test = pd.DataFrame(data_hl_feat_test, columns=columns)

def data_dtypes(data):
    for x in data.columns:
        if x == 'image_filename':
            data.loc[:, x] = data[x].astype(object)
        elif x == 'class_number':
            data.loc[:, x] = data[x].astype(int)
        else:
            data.loc[:, x] = data[x].astype(float)

data_dtypes(data_hl_feat_train)
data_dtypes(data_hl_feat_test)

# Save the transformed data
if not os.path.exists("./data/TransformedData/%s" % opt.model_type):
    os.mkdir("./data/TransformedData/%s" % opt.model_type)

path_hl_data = "./data/TransformedData/%s/%s_%s" % (opt.model_type, opt.model_name, opt.optimizer)
if not os.path.exists(path_hl_data):
    os.mkdir(path_hl_data)

data_hl_feat_train.to_csv(os.path.join(path_hl_data, "data_hl_feat_train.csv"), index=False)
data_hl_feat_test.to_csv(os.path.join(path_hl_data, "data_hl_feat_test.csv"), index=False)

# ============================ PLOT FEATURES DISTRIBUTION ============================ #
feature_names = [x for x in data_hl_feat_train.columns if x not in ['image_filename', 'class_number']]

X_hl_train = data_hl_feat_train.loc[:, feature_names].values
y_hl_train = data_hl_feat_train.loc[:, 'class_number'].values

colors = ['red','green','blue','orange']

if not os.path.exists("graphs/textural/hl_features/%s" % opt.model_type):
    os.mkdir("graphs/textural/hl_features/%s" % opt.model_type)

if not os.path.exists("graphs/textural/hl_features/%s/%s_%s" % (opt.model_type, opt.model_name, opt.optimizer)):
    os.mkdir("graphs/textural/hl_features/%s/%s_%s" % (opt.model_type, opt.model_name, opt.optimizer))

graph_path = "graphs/textural/hl_features/%s/%s_%s" % (opt.model_type, opt.model_name, opt.optimizer)

# DISTRIBUTION OF HIGH LEVEL FEATURES
if opt.plot_distrib:
    print("Plotting the distribution of each high level feature")
    for part in range(X_hl_train.shape[1]//12+1):
        fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
        ax = ax.flatten()
        for i in range(12):
            if i+12*part < X_hl_train.shape[1]:
                fname = feature_names[i+12*part]
                for l, c in zip(np.unique(y_hl_train), colors):
                    ax[i].hist(X_hl_train[y_hl_train==l, i+12*part], bins=100, density=True, alpha=0.5, color=c, label="cat %d" % l)
                    ax[i].set_title("Variable %s" % fname, fontweight="bold", fontsize=15)
                    ax[i].legend(loc="best", fontsize=10)
                    #ax[i].set_xlim([-5, 5])
            else:
                ax[i].axis('off')
        plt.suptitle("Distribution des variables high-level", fontsize=20, fontweight="bold")
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(graph_path +'/feat_distrib_%d.png' % part)
        plt.close('all')

# ============================ PCA ============================ #
# PROJECT
pca = PCA(n_components=X_hl_train.shape[1])
X_hl_train_pca = pca.fit_transform(X_hl_train)

expl_var =  pca.explained_variance_ratio_
cum_expl_var = np.cumsum(expl_var)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
ax.bar(np.arange(1, X_hl_train.shape[1]+1), expl_var, alpha=0.5,align='center',
        label='var. expl. individuelle')
ax.step(np.arange(1, X_hl_train.shape[1]+1), cum_expl_var, where='mid', label='var. expl. cumulée')
ax.set_xlabel('Indice composante principale', fontweight="bold", fontsize=20)
ax.set_ylabel('Ratio de variance expliquée',  fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=15)
plt.savefig(graph_path + '/pca_explained_var.png')

# PLOT PROJECTION ON PC 1 AND 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
colors = ['r','g','b','orange']
markers = ['+','x','o', '^']
for l, c, m in zip(np.unique(y_hl_train), colors,markers):
    ax.scatter(X_hl_train_pca[y_hl_train==l,0],
               X_hl_train_pca[y_hl_train==l,1],
               color=c,marker=m, label="cat %d" % l)

ax.set_xlabel('PC 1', fontweight="bold", fontsize=20)
ax.set_ylabel('PC 2', fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=20)
ax.set_title("Projection des variables de texture sur PC 1 et 2", fontweight="bold", fontsize=25)
plt.savefig(graph_path + '/pca_12.png')

# PLOT PROJECTION ON PC 3 AND 4
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
colors = ['r','g','b','orange']
markers = ['+','x','o', '^']
for l, c, m in zip(np.unique(y_hl_train), colors,markers):
    ax.scatter(X_hl_train_pca[y_hl_train==l,2],
               X_hl_train_pca[y_hl_train==l,3],
               color=c,marker=m, label="cat %d" % l)

ax.set_xlabel('PC 3', fontweight="bold", fontsize=20)
ax.set_ylabel('PC 4', fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=20)
ax.set_title("Projection des variables de texture sur PC 3 et 4", fontweight="bold", fontsize=25)
plt.savefig(graph_path + '/pca_34.png')

