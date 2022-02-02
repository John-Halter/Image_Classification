import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import keras
import tensorflow as tf
from Constants import LIST_OF_SPECIES, output_dir
from dataset_manipulation import create_training_data, create_testing_data
matplotlib.use('TkAgg')


# TODO: Fix error message with gpu not being used for tensorflow
batch_size = 32
img_height = 200
img_width = 200
training_dataset = create_training_data(batch_size,img_height,img_width, output_dir)
testing_dataset = create_testing_data(batch_size,img_height,img_width, output_dir)

class_names = training_dataset.class_names
print(class_names)



plt.show()