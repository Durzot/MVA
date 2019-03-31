# -*- coding: utf-8 -*-
"""
Merge predictions of a list of model on the test images and create submission file
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', nargs='+', type=str, default='pretrained_AlexNet_fz',  help='type of model')
opt = parser.parse_args()

# ======================= MERGE PREDICTIONS ======================== # 
df_merge = pd.read_csv('./data/test_data_order.csv', header='infer')

for model_type in opt.model_type:
    subs = os.listdir("submissions/%s" % model_type)
    for sub in subs:
        df = pd.read_csv("submissions/%s/%s" % (model_type, sub), header='infer')
        df_merge.loc[:, 'pred_%d' % i] = df.class_number.values

mask = [x for x in df_merge.columns if 'pred' in x]
df_merge.loc[:, 'class_number'] = df_merge.loc[:, mask].mode(axis=1).values[:, 0]

for x in df_merge.columns:
    if 'pred' in x:
        del df_merge[x]

df_merge.loc[:, 'class_number'] = df_merge.class_number.astype(int)

if not os.path.exists("./submissions/merge/"):
    os.mkdir("./submissions/merge")

df_merge.to_csv('submissions/merge/sub_%s.csv' % '_'.join(opt.model_type), header=True, index=False)


