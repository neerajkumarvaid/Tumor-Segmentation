#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 02:17:15 2019

@author: Neeraj Kumar

Description: This code converts any CNN with fully-connected layers
to a fully convolutional neural network and decouples the input to
take arbitrary sized inputs.
"""

import os
os.chdir('/home/vaid-computer/Neeraj/pcam/')
from keras.utils import HDF5Matrix
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#x_train = HDF5Matrix('camelyonpatch_level_2_split_train_x.h5', 'x')
#y_train = HDF5Matrix('camelyonpatch_level_2_split_train_y.h5', 'y')

import h5py

# Import TensorFlow and relevant Keras classes to setup the model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

#from tensorflow.keras.models import load_model
#model =  load_model('tumordetection_26July2019.h5')

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.engine import InputLayer
import keras

def to_fully_conv(model):

    new_model = tf.keras.Sequential()

    #input_layer = InputLayer(input_shape=(None, None, 3), name="input_new") # Use this for PyTorch
    input_layer = tf.keras.layers.InputLayer(input_shape = (None, None, 3),name="input_new") # Use this for Keras with Tensorflow backend
 
    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)
#                new_layer = Convolution2D(output_dim,
#                                          (f_dim[1],f_dim[2]),
#                                          strides=(1,1),
#                                          activation=layer.activation,
#                                          padding='valid',
#                                          weights=[new_W,b]) # Use this for PyTorch
                
                new_layer = tf.keras.layers.Conv2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])  # Use this for Keras with Tensorflow backend
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = tf.keras.layers.Conv2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])


        else:
            new_layer = layer

        new_model.add(new_layer)

    return new_model

#new_model = to_fully_conv(model)