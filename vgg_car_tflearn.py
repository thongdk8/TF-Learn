from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

from preprocessing_image import load_image

dir_train_folder = '/home/thongpb/TensorFlow_Work/data_car/CAM0_1_7'
dir_test_folder = '/home/thongpb/TensorFlow_Work/data_car/CAM1_1_7/test'

#dir_train_folder = '/home/nghia_VN/caffe/data/Car_pattern_Cam0'
#dir_test_folder = '/home/nghia_VN/caffe/data/Car_pattern_Cam1'

from tflearn.data_utils import image_preloader

# X, Y = image_preloader(files_list, image_shape=(224, 224), mode='file',
#                        categorical_labels=True, normalize=False,
#                        files_extension=['.jpg', '.png'], filter_channel=True)
# or use the mode 'floder'
X, Y = image_preloader(dir_train_folder, image_shape=(100, 100), mode='folder',
                       categorical_labels=True, normalize=True,
                       files_extension=['.jpg', '.png'], filter_channel=True)
test_x, test_y = image_preloader(dir_test_folder, image_shape=(100, 100), mode='folder',
                       categorical_labels=True, normalize=True,
                       files_extension=['.jpg', '.png'], filter_channel=True)





#X, Y = load_image(dir_train_folder=dir_train_folder)
#test_x, test_y = load_image(dir_test_folder=dir_test_folder, train_logical=False)


# Building 'VGG Network'
network = input_data(shape=[None, 100, 100, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)


network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 8, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.01)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet_car',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=10, validation_set=(test_x,test_y), shuffle=True,
          show_metric=True, batch_size=16, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_carmodel')