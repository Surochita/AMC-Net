#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 04:21:15 2020

@author: cssc
"""


import os
import math
import numpy as np
import nibabel as nib
from scipy import ndimage, misc
from scipy.ndimage.interpolation import zoom

import tensorflow as tf
import keras

from keras import backend as K
from keras import losses
from keras.models import Model, Sequential, load_model, model_from_json

import skimage
import random
import matplotlib.pyplot as plt
import cv2
 

initial_path = ""

def load_data_build_probability_kaggle():
    
    path = os.path.join(initial_path,'Kaggle/') 
    for sample in sorted(os.listdir(path)):
        
        name = sample.split('.')[0]
        
        # Read RGB image
        img = cv2.imread(path + sample)
        print(img.shape) #(512, 512)
        result = ndimage.generic_filter(img, np.nanmean, size=24, mode='constant', cval=np.NaN)
        
        cv2.imwrite('Kaggle/' +name + "_probability.png", result)
        
def load_data_build_probability_medseg1():
    
    path = os.path.join(initial_path,'Medseg1/') 
    for sample in sorted(os.listdir(path)):
        
        name = sample.split('.')[0]
        
        # Read RGB image
        img = cv2.imread(path + sample)
        print(img.shape) #(512, 512)
        result = ndimage.generic_filter(img, np.nanmean, size=24, mode='constant', cval=np.NaN)
        
        cv2.imwrite('Medseg1/' +name + "_probability.png", result)

def load_data_build_probability_medseg2():
    
    path = os.path.join(initial_path,'Medseg2/') 
    for sample in sorted(os.listdir(path)):
        
        name = sample.split('.')[0]
        
        # Read RGB image
        img = cv2.imread(path + sample)
        print(img.shape) #(512, 512)
        result = ndimage.generic_filter(img, np.nanmean, size=24, mode='constant', cval=np.NaN)
        
        cv2.imwrite('Medseg2/' +name + "_probability.png", result)

def load_data_build_probability_mosmed():
    
    path = os.path.join(initial_path,'Mosmed/') 
    for sample in sorted(os.listdir(path)):
        
        name = sample.split('.')[0]
        
        # Read RGB image
        img = cv2.imread(path + sample)
        print(img.shape) #(512, 512)
        result = ndimage.generic_filter(img, np.nanmean, size=20, mode='constant', cval=np.NaN)
        
        cv2.imwrite('Mosmed/' +name + "_probability.png", result)
    
if __name__ == '__main__':
    #load_data_build_probability_kaggle()
    #load_data_build_probability_medseg1()
    #load_data_build_probability_medseg2()
    load_data_build_probability_mosmed()
    

        
        
