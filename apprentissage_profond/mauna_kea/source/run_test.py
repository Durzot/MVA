# -*- coding: utf-8 -*-
"""
Run the model on the test images and prepare submission files
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
from dataset import *
from models import *
from utils import *

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--data_aug', type=int, default=0 , help='1 for data augmentation')
parser.add_argument('--n_classes', type=int, default=4, help='number of classes')
parser.add_argument('--model_path', type=str, default="trained_models/cnn_2/MaunaNet4_40.pth", help='dir where model is saved')
parser.add_argument('--model_type', type=str, default="cnn_2", help='model type')
parser.add_argument('--model_name', type=str, default="MaunaNet4",  help='optional reload model path')
parser.add_argument('--cuda', type=int, default=0, help='set to 1 to use cuda')
opt = parser.parse_args()

if not os.path.exists(opt.model_path):
    raise ValueError("There is no model %s" % opt.model_path)

# ========================== TEST DATA ========================== #
dataset_test_1 = MaunaKeaTest(root_img="./data/TestSetImagesDir/part_1", data_aug=opt.data_aug)
loader_test_1 = torch.utils.data.DataLoader(dataset_test_1, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

dataset_test_2 = MaunaKeaTest(root_img="./data/TestSetImagesDir/part_2", data_aug=opt.data_aug)
loader_test_2 = torch.utils.data.DataLoader(dataset_test_2, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

print('test set 1 size %d' % len(dataset_test_1))
print('test set 2 size %d' % len(dataset_test_2))

# ========================== NETWORK ========================== #
network = eval("%s(n_classes=opt.n_classes)" % opt.model_name)

if opt.cuda:
    network.load_state_dict(torch.load(opt.model_path))
    network.cuda()
else:
    network.load_state_dict(torch.load(opt.model_path), map_location='cpu')

print("Weights from %s loaded" % opt.model_path)

df_test = pd.DataFrame(columns=['image_filename', 'class_number'])
test_file = "./submissions/%s/sub_%s.csv" % (opt.model_type, opt.model_name)

# ====================== TESTING LOOP ====================== #
st_time = time.time()
network.eval()

for loader_test in [loader_test_1, loader_test_2]:
    with torch.no_grad():
        for fn, data in loader_test:
            if opt.cuda:
                data = data.cuda()
            
            output = network(data)
            pred = output.cpu().data.numpy().argmax(axis=1)
            
            for f, pr in zip(fn, pred):
                row_test = {'image_filename': f, 'class_number': pr}
                df_test = df_test.append(row_test, ignore_index=True)

dt = time.time()-st_time
print("%d min %d sec" % (dt//60, dt%60))
    
df_test.to_csv(test_file, header=True, index=False)

