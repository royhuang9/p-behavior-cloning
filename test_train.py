#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:54:47 2017

@author: roy
"""

from train import img_gen

import pandas as pd

import matplotlib.pyplot as plt

driving_info = pd.read_csv('/home/roy/data/debug_log.csv', skipinitialspace=True)
drv_dir = '/home/roy/data'

image_gen = img_gen(drv_dir, driving_info, 60)



X_train, y_train=next(image_gen)
X_train, y_train=next(image_gen)
X_train, y_train=next(image_gen)
X_train, y_train=next(image_gen)
X_train, y_train=next(image_gen)
X_train, y_train=next(image_gen)

'''

for img, y in zip(X_train, y_train):
    plt.figure(figsize=[12,12])
    plt.imshow(img.astype('uint8'))
    plt.title(str(y))
plt.show()
'''