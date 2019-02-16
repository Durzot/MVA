# -*- coding: utf-8 -*-
"""
Script for the competition by Mauna Kea Jan 2019 ENS 
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import sys
import numpy as np 
import pandas as pd

import PIL.Image as Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from

sys.path.append("./source")
from dataset import *

print(sys.version)
print(os.getcwd())
n_classes = 4

# ============================== VIZUALIZE IMAGES ============================== #
label_img = pd.read_csv("./data/TrainingSet_20aimVO.csv")
patient_col = label_img.image_filename.apply(lambda x: x.split("_")[-1].split(".")[0]).astype(int)
label_img.insert(0, column='patient', value=patient_col)    

root_img = "./data/TrainingSetImagesDir"
seed = 1995
np.random.seed(seed)

for cl in range(n_classes):
    mask = label_img.class_number == cl
    rand_rows = np.random.permutation(sum(mask))
    label_slct = pd.DataFrame.copy(label_img[mask].reset_index(drop=True))

    for n in range(10):
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(14, 14))
        ax = ax.flatten()
        label_shuffle = pd.DataFrame.copy(label_slct.iloc[rand_rows[n*16:(n+1)*16]]).reset_index(drop=True)
    
        for i, row in label_shuffle.iterrows():
            path = os.path.join(root_img, row["image_filename"])
            img = Image.open(path).convert('L')
            ax[i].imshow(img)
            ax[i].axis('off')
            ax[i].set_title("%s class %d" % (row["image_filename"], cl))
    
        fig.savefig("graphs/class_%d/samples_class_%d_part_%d.png" % (cl, cl, n), format="png")
        plt.close(fig)
