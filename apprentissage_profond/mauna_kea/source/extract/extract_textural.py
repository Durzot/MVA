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
import datetime

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

sys.path.append("./source")
from auxiliary.dataset import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========================== EXTRACT TEXTURAL FEATURES DATA ========================== #
dataset_train = MaunaTexturalFeatures(train=True, feature_mean=True, feature_std=True, feature_range=True)
data_textural_train =  dataset_train.get_data()
data_textural_train.to_csv('./data/TransformedData/data_textural_train.csv', index=False)

dataset_test_1 = MaunaTexturalFeatures(train=False, root_img="./data/TestSetImagesDir/part_1", feature_mean=True, feature_std=True, feature_range=True)
data_textural_test_1 =  dataset_test_1.get_data()
data_textural_test_1.to_csv('./data/TransformedData/data_textural_test_1.csv', index=False)

dataset_test_2 = MaunaTexturalFeatures(train=False, root_img="./data/TestSetImagesDir/part_2", feature_mean=True, feature_std=True, feature_range=True)
data_textural_test_2 =  dataset_test_2.get_data()
data_textural_test_2.to_csv('./data/TransformedData/data_textural_test_2.csv', index=False)

feature_cols = [x for x in data_textural_train.columns if x not in ['patient', 'image_filename', 'label']]

X = data_textural_train.loc[:, feature_cols].values
y = data_textural_train.loc[:, 'class_number'].astype(int).values

# Normalize the data
sc = StandardScaler()
X_std = sc.fit_transform(X)

# ============================ PLOT FEATURES DISTRIBUTION ============================ #
colors = ['red','green','blue','orange']

# DISTRIBUTION OF 6 FIRST FEATURES
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
ax = ax.flatten()
for i in range(6):
    fname = feature_cols[i]
    for l, c in zip(np.unique(y), colors):
        ax[i].hist(X_std[y==l, i], bins=100, density=True, alpha=0.5, color=c, label="cat %d" % l)
        ax[i].set_title("Variable %s" % fname, fontweight="bold", fontsize=15)
        ax[i].legend(loc="best", fontsize=12)
        ax[i].set_xlim([-5, 5])
plt.suptitle("Distribution des variables de texture normalisées", fontsize=20, fontweight="bold")
plt.savefig('graphs/textural/traditional_features/feat_distrib_1.png')

# DISTRIBUTION OF 6 LAST FEATURES
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
ax = ax.flatten()
for i in range(6):
    fname = feature_names[i+6]
    for l, c in zip(np.unique(y), colors):
        ax[i].hist(X_std[y==l, i+6], bins=100, density=True, alpha=0.5, color=c, label="cat %d" % l)
        ax[i].set_title("Variable %s" % fname, fontweight="bold", fontsize=15)
        ax[i].legend(loc="best", fontsize=12)
        ax[i].set_xlim([-5, 5])
plt.suptitle("Distribution des variables de texture normalisées", fontsize=20, fontweight="bold")
plt.savefig('graphs/textural/traditional_features/feat_distrib_2.png')

# ============================ PCA ============================ #
cov_mat = np.cov(X_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

tot = sum(eigen_vals)
expl_var = [i/tot for i in sorted(eigen_vals,reverse=True)]
cum_expl_var = np.cumsum(expl_var)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
ax.bar(np.arange(1, X_std.shape[1]+1), expl_var,alpha=0.5,align='center',
        label='var. expl. individuelle')
ax.step(np.arange(1, X_std.shape[1]+1), cum_expl_var, where='mid', label='var. expl. cumulée')
ax.set_xlabel('Indice composante principale', fontweight="bold", fontsize=20)
ax.set_ylabel('Ratio de variance expliquée',  fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=15)
plt.savefig('graphs/textural/traditional_features/pca_explained_var.png')

# PROJECT
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_std)

# PLOT PROJECTION ON PC 1 AND 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
colors = ['r','g','b','orange']
markers = ['+','x','o', '^']
for l, c, m in zip(np.unique(y), colors,markers):
    ax.scatter(X_pca[y==l,0],
               X_pca[y==l,1],
               color=c,marker=m, label="cat %d" % l)

ax.set_xlabel('PC 1', fontweight="bold", fontsize=20)
ax.set_ylabel('PC 2', fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=20)
#ax.set_xlim([-10, 10])
#ax.set_ylim([-10, 10])
ax.set_title("Projection des variables de texture sur PC 1 et 2", fontweight="bold", fontsize=25)
plt.savefig('graphs/textural/traditional_features/pca_12.png')

# PLOT PROJECTION ON PC 3 AND 4
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
colors = ['r','g','b','orange']
markers = ['+','x','o', '^']
for l, c, m in zip(np.unique(y), colors,markers):
    ax.scatter(X_pca[y==l,2],
               X_pca[y==l,3],
               color=c,marker=m,label="cat %d" % l)

ax.set_xlabel('PC 3', fontweight="bold", fontsize=20)
ax.set_ylabel('PC 4', fontweight="bold", fontsize=20)
ax.legend(loc='best', fontsize=20)
ax.set_title("Projection des variables de texture sur PC 1 et 2", fontweight="bold", fontsize=25)
plt.savefig('graphs/textural/traditional_features/pca_34.png')

