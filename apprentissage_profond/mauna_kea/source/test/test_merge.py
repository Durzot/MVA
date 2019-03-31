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
from scipy import stats

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', nargs='+', type=str, default='pretrained_AlexNet_fz',  help='type of model')
opt = parser.parse_args()

# ======================= MERGE PREDICTIONS ======================== # 
df_merge = pd.read_csv('./data/test_data_order.csv', header='infer')

for model_type in opt.model_type:
    subs = os.listdir("submissions/%s" % model_type)
    for sub in subs:
        if ".csv" in sub:
            df = pd.read_csv("submissions/%s/%s" % (model_type, sub), header='infer')
            df_merge.loc[:, 'pred_%s' % sub] = df.class_number.values

mask = [x for x in df_merge.columns if 'pred' in x]
df_merge.loc[:, 'class_number'] = df_merge.loc[:, mask].mode(axis=1).values[:, 0]

for x in df_merge.columns:
    if 'pred' in x:
        del df_merge[x]

df_merge.loc[:, 'class_number'] = df_merge.class_number.astype(int)

if not os.path.exists("./submissions/merge/"):
    os.mkdir("./submissions/merge")

df_merge.to_csv('submissions/merge/sub_%s.csv' % '_'.join(opt.model_type), header=True, index=False)

#################################
# Predict only 1 label per patient
#################################

df_patient = pd.DataFrame(columns=["patient", "image_filename", "class_number"])
for _, row in df_merge.iterrows():
    pat_row = {'patient': row['image_filename'].split(".")[0].split("_")[-1], 
               'image_filename': row['image_filename'],
               'class_number': int(row['class_number'])}
    df_patient = df_patient.append(pat_row, ignore_index=True)
df_patient.loc[:, 'class_number'] = df_patient.class_number.astype(int)

# Select most frequent predicted label per patient
s_patient = df_patient[['patient', 'class_number']].groupby('patient').agg(lambda x: stats.mode(x)[0][0])     

for row in s_patient.iterrows():
    df_patient.loc[df_patient.patient == row[0], 'class_number'] = row[1]['class_number']
     
df_patient[['image_filename', 'class_number']].to_csv('submissions/merge/sub_%s_patient.csv' % '_'.join(opt.model_type), header=True, index=False)

