#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:46:23 2020

@author: cssc
"""

import numpy as np
np.random.seed(1337) # for reproducibility

import tensorflow as tf
import keras

from keras import backend as K
from keras import losses

from keras.models import Model, Sequential
from keras.objectives import categorical_crossentropy

from keras.activations import sigmoid

from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K

from keras.optimizers import Adam, Adadelta, Adamax, Nadam, Adagrad, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback

from keras_preprocessing.image import ImageDataGenerator

from keras.initializers import RandomUniform, RandomNormal

from keras.regularizers import l2


#model block implementation
def attention_up_and_concate(down_layer, layer):
    data_format = 'channels_last'
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    out_channel = in_channel*2
    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    # up = UpSampling2D(size=(2, 2))(down_layer) // do Conv2DTranspose, written in upper line

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x



def MS_block(inputs, features, kernel=(3, 3), strides=(1, 1)):
    x = inputs
    
    array = [1, 2, 3, 5] # dilation rates
    
    #block 1
    b1 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[0])(x)
    #b1 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[0])(b1)
    
    #block 2
    b2 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[1])(b1)
    #b2 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[1])(b2)
    
    #block 3
    b3 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[2])(b2)
    #b3 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[2])(b3)
    #b3 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[2])(b3)
    
    #block 4
    b4 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[3])(b3)
    #b4 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[3])(b4)
    #b4 = Conv2D(features, (3, 3), activation='relu', padding='same', dilation_rate=array[3])(b4)
    
    c = concatenate([b1, b2, b3, b4], axis=3)
    drop1 = Dropout(0.2)(c)
    conv1 = Conv2D(features, (1, 1), activation='relu', padding='same')(drop1)
    drop2 = Dropout(0.2)(conv1)
    
    return drop2

    


def MS_att_unet(optimizer, loss_metric, metrics, lr=1e-3):
    inputs = Input((128, 128, 1))
    #print("Input Shape: \t", inputs.shape)
    x = inputs
    depth = 4
    features = 16
    skips = []
    
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        #print("After 1st Convolution: \t", x.shape)
        x = Dropout(0.2)(x)
        #print("After Dropout: \t", x.shape)
        x = MS_block(x, features)
        #print("After 2nd Convolution: \t", x.shape)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        #print("After Maxpooling: \t", x.shape)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
    #print("After 1st Convolution: \t", x.shape)
    x = Dropout(0.2)(x)
    #print("After Droprout: \t", x.shape)
    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
    #print("After 2nd Convolution: \t", x.shape)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i])
        #print("After Attention: \t", x.shape)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        #print("After 1st Convolution: \t", x.shape)
        x = Dropout(0.2)(x)
        #print("After Dropout: \t", x.shape)
        x = x = MS_block(x, features)
        #print("After 2nd Convolution: \t", x.shape)

    conv6 = Conv2D(1, (1, 1), padding='same')(x)
    #print("After last Convolution: \t", conv6.shape)
    conv7 = core.Activation('sigmoid')(conv6)
    #print("After Activation: \t", conv7.shape)
    model = Model(inputs=[inputs], outputs=conv7)

    model.compile(optimizer=optimizer(lr=lr, decay=1e-6, clipvalue=0.5), loss=loss_metric, metrics=metrics)
    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model




