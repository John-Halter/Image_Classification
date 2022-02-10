import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential

from dataset_gather import download_pictures, verify_pictures
from Constants import LIST_OF_SPECIES, output_dir
from dataset_manipulation import create_training_data

matplotlib.use('TkAgg')

download_pics = False
num_of_pics = 100
verify_pics = True
run_cnn = False

if download_pics:
  for i in range(len(LIST_OF_SPECIES)):
      download_pictures(LIST_OF_SPECIES[i], num_of_pics)

if verify_pics:
  verify_pictures(output_dir)

# TODO: Fix error message with gpu not being used for tensorflow
if run_cnn:
    img_height = 200
    img_width = 200
    training_dataset = create_training_data(img_height, img_width, output_dir, 'training')
    testing_dataset = create_training_data(img_height, img_width, output_dir, 'validation')

    class_names = training_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = testing_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # What NN looks like
    # All training data has to be the same size! See that the last sequence is the number of species
    model = Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    # Training done here
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())

    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
