#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:27:29 2017

@author: roy
"""
import argparse
import numpy as np
import json
import os
import sys

from PIL import Image

from keras.models import model_from_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict image')
    parser.add_argument('--model', type=str, help='Model file')
    parser.add_argument('--image', type=str, help='image file for predict')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print('{} is not exist'.format(args.model))
        sys.exit(-1)

    print(args.model)
    
    if not os.path.exists(args.image):
        print('{} is not exist'.format(args.image))
        sys.exit(-1)

    print(args.image)

    with open(args.model, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))
    
    model.compile('adam', 'mse')
    
    weights_file = args.model.replace('json','h5')
    
    model.load_weights(weights_file)
    
    image = np.asarray(Image.open(args.image))
    images = image[None, :,:,:]
    
    pred_value = model.predict(images,1)
    print('predict type: {}, value: {}'.format(type(pred_value), pred_value))
    