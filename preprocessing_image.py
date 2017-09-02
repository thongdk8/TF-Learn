import cv2
import numpy as np
import os
import sys


def to_arr_target(labels):
	temp = np.zeros(8)
	temp[labels-1] = 1
	return temp


def preprocess_image(path):
	readImage = cv2.imread(path)
	readImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
	readImage = cv2.resize(readImage,(100,100))
	image_arr = np.asarray(readImage)
	image_arr = np.array(image_arr)
	element_folder = str(path).split('/')
	label = int(element_folder[-2])

	return image_arr, label

def load_image(dir_train_folder='',dir_test_folder='',train_logical=True):
	if train_logical:
		files = []
		for i in range(1, 9):
			fileDir = os.path.join(dir_train_folder,'{}'.format(str(i)))
			for name in os.listdir(fileDir):
				if os.path.isfile(os.path.join(fileDir, name)):
					files.append(os.path.join(fileDir, name))
	else:
		files = []
		for i in range(1, 9):
			fileDir = os.path.join(dir_test_folder,'{}'.format(str(i)))
			for name in os.listdir(fileDir):
				if os.path.isfile(os.path.join(fileDir, name)):
					files.append(os.path.join(fileDir, name))

	images = []
	labels = []
	number_of_image = len(files)
	for i in range(0,number_of_image):
		t_image, t_label = preprocess_image(files[i])
		images.append(t_image)
		labels.append(t_label)
	images = np.array([x for x in images])
	labels = np.array([to_arr_target(x) for x in labels])
	print('Shape image is ',np.shape(images))
	print('Shape label is ',np.shape(labels))
	return (images, labels)