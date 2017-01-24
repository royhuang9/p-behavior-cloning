#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:33:15 2017

@author: roy
"""

import pickle


with open('./loss/10000_log.loss', 'rb') as fp:
    hist = pickle.load(fp)
    print(hist['val_loss'][50:100])
