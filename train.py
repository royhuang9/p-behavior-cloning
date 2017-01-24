#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:11:37 2017

@author: roy
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import pickle
from PIL import Image

#np.random.seed(8888)  # for reproducibility

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

epoch_size = 0
batch_size = 50 * 6
nb_epoch = 1

# left and right image will be used.
# all image will be horizontal reverted.
# so we got 6 time more than only center image
times = 6
center_only = 0

# input image dimensions
img_ch, img_rows, img_cols = 3, 160, 320

model = None

# split the data set into train and validation according to rate
def split_data(dataset, rate):
    msk = np.random.rand(len(dataset)) < rate
    train_part = dataset[msk].reset_index(drop=True)
    val_part = dataset[~msk].reset_index(drop=True)
    return train_part, val_part

# return a value between 0.1 and 0.2
def shift_angle():
    return np.random.ranf() * 0.1 + 0.1
# image generator
def img_gen(driving_dir, driving_info, b_size):
    total_images=driving_info.shape[0]

    driving_info = driving_info.sample(frac=1).reset_index(drop=True)
    features = np.empty((b_size, img_rows, img_cols, img_ch), dtype=np.float32)
    output = np.empty(b_size, dtype = np.float32)

    b_step = b_size // times
    pos_start = 0
    pos_end = pos_start + b_step
    while True:
        left = 0
        if pos_end > total_images:
            left = pos_end - total_images
            pos_end = total_images
        
        for idx, pos in enumerate(range(pos_start, pos_end)):
            # get the center image
            
            file_name = driving_info.ix[pos]['center']
            file_full = os.path.join(driving_dir, file_name)
            #print(file_full, end=', ')
            #print(file_full)
            img_pil = Image.open(file_full)
            features[idx * times] = np.array(img_pil)
            
            steering_angle = float(driving_info.ix[pos]['steering'])
            #features[idx] = np.array(Image.open(file_name))
            output[idx * times] = steering_angle
            #print(output[idx * times], end=', ')
            
            # reverse center image and steering angle
            img_flip = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            features[idx * times + 1] = np.array(img_flip)
            output[idx * times + 1] = 0 - steering_angle
            #print(output[idx * times + 1])
            
            if center_only == True:
                continue
            
            # get the right image
            file_name = driving_info.ix[pos]['right']
            file_full = os.path.join(driving_dir, file_name)
            #print(file_full, end=', ')
            img_pil = Image.open(file_full)
            features[idx * times + 2] = np.array(img_pil)
            output[idx * times + 2] = steering_angle - shift_angle()
            #print(output[idx * times + 2], end=', ')
            
            # reverse the right image
            img_flip = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            features[idx * times + 3] = np.array(img_flip)
            output[idx * times + 3] = steering_angle + shift_angle()
            #print(output[idx * times + 3])
            
            # get the left image
            file_name = driving_info.ix[pos]['left']
            file_full = os.path.join(driving_dir, file_name)
            #print(file_full, end=', ')
            img_pil = Image.open(file_full)
            features[idx * times + 4] = np.array(img_pil)
            output[idx * times + 4] = steering_angle + shift_angle()
            #print(output[idx * times + 4], end=', ')
            
            # reverse the left image
            img_flip = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            features[idx * times + 5] = np.array(img_flip)
            output[idx * times + 5] = steering_angle  - shift_angle()
            #print(output[idx * times + 5])

        res_features = features[0:b_size - left * times]
        res_output = output[0:b_size - left * times]
        
        #print('{} images gen'.format(len(res_features) ))
        
        yield res_features, res_output
        
        pos_start = 0 if left > 0 else pos_start + b_step
        pos_end = pos_start + b_step

# number of convolutional filters to use
nb_filters = {
              'conv1':24,
              'conv2':36,
              'conv3':48,
              'conv4':64,
              'conv5':64,
              'fc1':100,
              'fc2':50,
              'fc3':10}
              
def get_model():
    print('Create new model')
    model = Sequential()
  
    # layer 0 : normalization
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(img_rows, img_cols, img_ch),
                     output_shape=(img_rows, img_cols, img_ch)))

    # layer 1, convnet 1, input: 160 x 320 x 3, output 40 x 80 x 24
    model.add(Convolution2D(nb_filters['conv1'], 5, 5,
                            activation = 'relu',
                            W_regularizer = l2(0.001), 
                            subsample=(2, 2), 
                            border_mode="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())

    # layer 2, convnet 2, input: 40 x 80 x 24, output: 20 x 40 x 36
    model.add(Convolution2D(nb_filters['conv2'], 5, 5, 
                            activation='relu',
                            W_regularizer = l2(0.001), 
                            subsample=(2, 2),
                            border_mode="same"))
    model.add(BatchNormalization())

    # layer 3, convnet 3, input: 20 x 40 x 36, output: 10 x 20 x 48
    model.add(Convolution2D(nb_filters['conv3'], 5, 5,
                            activation='relu',
                            W_regularizer = l2(0.001),
                            subsample=(2, 2),
                            border_mode="same"))
    model.add(BatchNormalization())

    # layer 4, convnet 4, input: 10 x 20 x 48, output: 5 x 10 x 64
    model.add(Convolution2D(nb_filters['conv4'], 3, 3, 
                            activation='relu',
                            W_regularizer = l2(0.001),
                            subsample=(2, 2),
                            border_mode="same"))
    model.add(BatchNormalization())

    # layer 5, convnet 5, input: 5 x 10 x 64, output: 3 x 8 x 64
    model.add(Convolution2D(nb_filters['conv5'], 3, 3,
                            activation='relu',
                            W_regularizer = l2(0.001),
                            border_mode="valid"))
    model.add(BatchNormalization())

    # layer 6, full connected 1
    model.add(Flatten())
    model.add(Dense(nb_filters['fc1'], W_regularizer = l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())

    # layer 7, full connected 2
    model.add(Dense(nb_filters['fc2'], W_regularizer = l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())

    # layer 8, full connected 3
    model.add(Dense(nb_filters['fc3'], W_regularizer = l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    return model

def load_model(args):
    print('load the mode:{}'.format(args.model))
    
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))
  
    opt_adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=opt_adam)
    
    weights_file = args.model.replace('json','h5')
    
    model.load_weights(weights_file)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--drv_file', type=str, help='driving log file')
    parser.add_argument('--val_file', type=str, help='validation log file')
    parser.add_argument('--epoch', type=int, default=2, help='Number of epochs.')
    parser.add_argument('--load', type=int, default=0, help='load old data to refine')
    parser.add_argument('--model', type=str, help='an old model json file')
    parser.add_argument('--rate', type=float, default=0.7, help='rate for training')
    parser.add_argument('--center', type=int, default=0, help='refine tune mode')

    args = parser.parse_args()

    if not os.path.exists(args.drv_file):
        print('{} is not exist'.format(args.drv_file))
        sys.exit(-1)
    print(args.drv_file)

    '''    
    if not os.path.exists(args.val_file):
        print('{} is not exist'.format(args.val_file))
        sys.exit(-1)
    print(args.val_file)
    '''
    
    driving_info = pd.read_csv(args.drv_file, skipinitialspace=True)
    drv_dir = os.path.dirname(args.drv_file)
    
    train_part, val_part = split_data(driving_info, args.rate)
    
    center_only = args.center
    times = 6 if center_only == 0 else 2
    
    epoch_size = train_part.shape[0] * times
    val_size = val_part.shape[0] * times
    
    nb_epoch = args.epoch
    
    print('epoch_size:{}, val_size:{}, nb_epoch:{}'.format(epoch_size, val_size, nb_epoch))
    #print('epoch_size:{}, nb_epoch:{}'.format(epoch_size, nb_epoch))
    if args.load == 0:
        model = get_model()
        model.summary()
    else:
        model = load_model(args)

    print("Saving model weights and configuration file.")
    with open('/home/roy/data/models/model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    # store weight every epoch
    file_path = '/home/roy/data/models/weights.{epoch:03d}-{val_loss:.4f}.hdf5'
    save_model = ModelCheckpoint(file_path, verbose=1)
    # When the validation loss doesn't decrease, just stop training
    stop_it = EarlyStopping(min_delta=0.002, patience=5, verbose=1)
    all_cbs = [stop_it, save_model]
    hist = model.fit_generator(
                        img_gen(drv_dir, train_part, batch_size),
                        samples_per_epoch = epoch_size,
                        nb_epoch = nb_epoch,
                        validation_data = img_gen(drv_dir, val_part, batch_size),
                        nb_val_samples = val_size,
                        callbacks=all_cbs
                        )
    

    
    loss_file = os.path.basename(args.drv_file).replace('csv', 'loss')
    loss_file=os.path.join('./loss/', loss_file)
    
    print(loss_file)
    with open(loss_file, 'wb') as lf:
        pickle.dump(hist.history, lf, pickle.HIGHEST_PROTOCOL)
    
        
