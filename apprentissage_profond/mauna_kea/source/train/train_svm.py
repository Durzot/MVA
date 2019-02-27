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

sys.path.append("./source")
from auxiliary.dataset import *
from auxiliary.utils import *

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.svm import SVC

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M") 

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/TransformedData/TrainTransformed.csv', help='path data')
parser.add_argument('--n_classes', type=int, default=4, help='number of classes')
parser.add_argument('--model_type', type=str, default='svm_1',  help='type of model')
parser.add_argument('--model_name', type=str, default='SVM',  help='name of the model for log')
parser.add_argument('--n_splits', type=int, default=5, help='number of splits for cv of params')
parser.add_argument('--test_size', type=float, default=0.2, help='size of held-out data')
parser.add_argument('--random_state', type=int, default=0, help='random state for the split of data')
opt = parser.parse_args()

# ========================== TRAINING DATA ========================== #
if opt.data_path is None:
    dataset = MaunaTexturalFeatures()
    data_textural =  dataset.get_data()
else:
    data_textural = pd.read_csv(opt.data_path, header='infer')

X = data_textural.loc[:, [x for x in data_textural.columns if x not in ['patient', 'label']]].values
y = data_textural.loc[:, 'label'].astype(int).values

# ========================== DATA SPLIT ========================== #
# Build train/test splits at patient level
kf = KFold(n_splits=opt.n_splits, shuffle=True, random_state=0)
cv_splits = []
for train_pat, test_pat in kf.split(data_textural.patient.unique()):
    train_idx = data_textural.loc[data_textural.patient.isin(train_pat)].index
    test_idx = data_textural.loc[data_textural.patient.isin(test_pat)].index
    cv_splits.append((train_idx, test_idx))

# ========================== MODEL AND GRIDSEARCH ========================== #
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


