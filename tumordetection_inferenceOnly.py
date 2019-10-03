#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:43:54 2019

@author: Neeraj Kumar

Description: Performs tumor segmentation on whole slide images of tissue sections

Requirements: 
    
    Weights: tumordetection_26July2019.h5
    Code: to_fcn.py 
    Libraries: Keras, Tensorflow, Numpy, Openslide, PIL, Scipy
"""

import os
os.chdir('/home/vaid-computer/Neeraj/pcam')
import numpy as np
import scipy.io as sio
# Load the saved model run it on the test set
from tensorflow.keras.models import load_model
import glob
#Process whole slide images
import openslide
import to_fcn
from PIL import Image


# Load the saved model run it on the test set
model =  load_model('tumordetection_26July2019.h5')

# Convert the model to the fully convolutional one
model = to_fcn.to_fully_conv(model)

# Read in the image
test_data_dir = '/media/vaid-computer/NeerajHDWindows/CCIPD/Willis_May/'
# Create a list of .svs or .tif tissue images
wsi_paths = glob.glob(os.path.join(test_data_dir, '*.tif'))
wsi_paths.sort()
WSI_path = list(wsi_paths)

case_num = 895
for img_num in range(1):
    image_name = WSI_path[img_num]

    img = openslide.OpenSlide(image_name)
    # Check the minimum resolution level of your images before running this
    level = img.level_count-2 # Process the image saved at leat resolution in svs file, use 0 for highest resolution image
    print(f"Processing image # {case_num}")
    
    im = np.array(img.read_region((0,0),level,img.level_dimensions[level]).convert('RGB'))
    im = im.astype('float32')
    im = im/255.
    im = np.expand_dims(im,axis = 0)
    tumor = model.predict(im)
    tumor = np.squeeze(tumor)[:,:,1]
    
    # Find tumor regions with probability at least 0.95
    tumor = tumor > 0.95
    # Convert numpy array to PIL image
    tim = Image.fromarray(np.uint8(tumor))
    # Resize the tumor image to the level-th resolution (x,y- flipped)
    # Didn't write the deconvolution and upsampling leg of FCN because for tumor segmentation its not required
    # We can quickly upsample the lost pixels to convolutions
    tim = tim.resize((img.level_dimensions[level][0],img.level_dimensions[level][1]))
    
    save_name = image_name[:54] + 'Tumor_probabilities/' + image_name[54:54+12] + '_tumor_prob.mat'
    sio.savemat(save_name,{'tumor_prob':tumor})
    case_num += 1
    
    