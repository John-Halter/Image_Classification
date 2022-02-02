import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import keras
import tensorflow as tf
from Constants import LIST_OF_SPECIES, output_dir
matplotlib.use('TkAgg')


def create_training_data():
    for species in LIST_OF_SPECIES:
        path = os.path.join(output_dir, species + " Butterfly")
        class_num = LIST_OF_SPECIES.index(species)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR) # Shape is size (some number, some number, 3) 3 for rgb
            plt.imshow(img_array)
            plt.show()
            break
        break



# model = tf.keras.models.Sequential(
#     tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (180,180,3)
# ))

plt.show()