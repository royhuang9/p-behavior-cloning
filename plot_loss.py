#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 21:11:00 2017

@author: roy
"""

import matplotlib.pyplot as plt
import os
import pickle

file_dir = './loss/'

filelist = os.listdir(file_dir)

for one_file in filelist:
    full_name = os.path.join(file_dir, one_file)
    with open(full_name, 'rb') as fp:
        hist = pickle.load(fp)
        plt.figure(figsize=(12,12))
        plt.plot(hist['loss'], 'r')
        plt.plot(hist['val_loss'], 'g')
        plt.set_xlim=([0,400])
        plt.set_ylim=([0, 1])
        plt.title(os.path.basename(full_name))
        
plt.show()