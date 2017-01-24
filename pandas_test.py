#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:52:30 2017

@author: roy
"""
import os
import pandas as pd

path_dir= '/home/roy/classes/simulator/data_collect/data'

driving_info = pd.read_csv(os.path.join(path_dir, 'driving_log.csv'))

one_object = driving_info.ix[1]
print(type(one_object))
print(driving_info.ix[1]['steering'])
print(driving_info.shape)