

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from preprocessing_image import load_image
import tensorflow as tf
import numpy as np
import cv2

dir_train_folder = '/home/thongpb/TensorFlow_Work/data_car/CAM0_1_7'
#dir_test_folder = '/home/thongpb/TensorFlow_Work/data_car/CAM1_1_7/test'

#dir_train_folder = '/home/nghia_VN/caffe/data/Car_pattern_Cam0'
#ir_test_folder = '/home/nghia_VN/caffe/data/Car_pattern_Cam1'

#from tflearn.data_utils import image_preloader

#X, Y = image_preloader(dir_train_folder, image_shape=[100,100], mode='folder',
#           files_extension=['.jpg', '.png'])

X, Y = load_image(dir_train_folder=dir_train_folder)
#test_x, test_y = load_image(dir_test_folder=dir_test_folder, train_logical=False)






# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Building 'AlexNet'
network = input_data(shape=[None, 100, 100, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 8, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.01)

# Training
model = tflearn.DNN(network, tensorboard_dir='./log')
model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=16, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_carmodel')
model.save('alex_carmodel.tfmodel')