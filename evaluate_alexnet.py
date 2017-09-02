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

#dir_train_folder = '/home/thongpb/TensorFlow_Work/data_car/CAM0_1_7'
dir_test_folder = '/home/thongpb/TensorFlow_Work/data_car/CAM1_1_7_full'

#dir_train_folder = '/home/nghia_VN/caffe/data/Car_pattern_Cam0'
#dir_test_folder = '/home/nghia_VN/caffe/data/Car_pattern_Cam1'

from tflearn.data_utils import image_preloader

# X, Y = image_preloader(files_list, image_shape=(224, 224), mode='file',
#                        categorical_labels=True, normalize=False,
#                        files_extension=['.jpg', '.png'], filter_channel=True)
# or use the mode 'floder'
# X, Y = image_preloader(dir_test_folder, image_shape=(100, 100), mode='folder',
#                        categorical_labels=True, normalize=False,
#                        files_extension=['.jpg', '.png'], filter_channel=True)
# test_x, test_y = image_preloader(dir_test_folder, image_shape=(100, 100), mode='folder',
#                        categorical_labels=True, normalize=False,
#                        files_extension=['.jpg', '.png'], filter_channel=True)


test_x, test_y = load_image(dir_test_folder=dir_test_folder,
						 train_logical=False)

num_sample = np.shape(Y)[0]
print('Testing on {} sample'.format(num_sample))



#X, Y = load_image(dir_train_folder=dir_train_folder)
#


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
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_dir='./log')
model.load('alex_carmodel.tfmodel')

#model.evaluate(X, Y, batch_size=128 )

#print('Prediction of X[0]: ', model.predict( [ X[0] ] ) )
correct_num = 0
i=0

while True:
	if(i>num_sample):
		break
	predict = np.argmax(model.predict(X[i:(i+100)]),1) 
	original =  np.argmax(Y[i:(i+100)],1)
	correct_num += np.sum(np.equal(predict,original))
	i += 100
	if(i%500 == 0):
		print('Current generation: #{} '.format(i))
print('Accuracy: {:.3f}'.format(correct_num/num_sample))