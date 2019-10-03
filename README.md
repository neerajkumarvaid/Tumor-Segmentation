# Tumor-Segmentation
This repository contains an implementation of a deep learning algorithm for tumor segmentation from Whole Slide Images of tissue sections.

Requirements: 
    
    Weights: tumordetection_26July2019.h5
    Code: to_fcn.py 
    Libraries: Keras, Tensorflow, Numpy, Openslide, PIL, Scipy
    
How to use?

Tumordetection_inferenceOnly.py is the main file that will read in the weights and supporting code to run the tumor segmentation model. You can change the directory of your images at line 37 of this file. 
