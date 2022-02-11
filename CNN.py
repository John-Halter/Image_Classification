import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from pathlib2 import Path

from dataset_gather import download_pictures, verify_pictures, split_train_test
from Constants import LIST_OF_SPECIES, OUTPUT_DIR, IMG_WIDTH, IMG_HEIGHT
from dataset_manipulation import create_training_data

matplotlib.use('TkAgg')

download_pics = False
num_of_pics = 100
verify_pics = False
split = False

run_cnn = True


if split:
    split_train_test(OUTPUT_DIR)


if download_pics:
  for i in range(len(LIST_OF_SPECIES)):
      download_pictures(LIST_OF_SPECIES[i], num_of_pics)

if verify_pics:
  verify_pictures(OUTPUT_DIR)

# TODO: Fix error message with gpu not being used for tensorflow
if run_cnn:


    # training_dataset = create_training_data(IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIR, 'training')
    # testing_dataset = create_training_data(IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIR, 'validation')
    #
    # data_augmentation = keras.Sequential(
    #     [
    #         tf.keras.layers.RandomFlip("horizontal",input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #         tf.keras.layers.RandomRotation(0.2),
    #         tf.keras.layers.RandomZoom(0.2),
    #     ]
    # )


    training_dataset = Path.cwd() / 'train'
    testing_dataset = Path.cwd() / 'test'

    train_datagen = ImageDataGenerator(
        rotation_range= 30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False)
    test_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False)
    training_datagen = train_datagen.flow_from_directory(
        training_dataset,
        target_size=(200, 200),
        batch_size=32,
        class_mode="categorical")
    testing_datagen = test_datagen.flow_from_directory(
        testing_dataset,
        target_size=(200, 200),
        batch_size=32,
        class_mode="categorical")



    class_names = training_datagen.class_indices # class_indices
    num_classes = len(class_names)

    # AUTOTUNE = tf.data.AUTOTUNE
    #
    # train_ds = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = testing_dataset.cache().prefetch(buffer_size=AUTOTUNE)



    # What NN looks like
    # All training data has to be the same size! See that the last sequence is the number of species
    # Model is overfitting
    model = Sequential([
        # data_augmentation,
        tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes,activation=tf.nn.softmax)
    ])

    # Training done here
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(), # CategoricalCrossentropy
                  metrics=['accuracy'])

    print(model.summary())

    epochs = 30
    history = model.fit(
        training_datagen,
        validation_data=testing_datagen,
        epochs=epochs,
        verbose=2
    )




    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    fig, ax = plt.subplots(2,figsize=(13, 8))
    ax[0].plot(epochs_range, acc, label='Training Accuracy')
    ax[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    ax[0].set_ylabel("Accuracy Percentage")
    ax[0].legend(loc='lower right')
    ax[0].set_title('Training and Validation Accuracy')


    ax[1].plot(epochs_range, loss, label='Training Loss')
    ax[1].plot(epochs_range, val_loss, label='Validation Loss')
    ax[1].set_xlabel("Number of Epochs")
    ax[1].legend(loc='upper right')
    ax[1].set_title('Training and Validation Loss')
    plt.show()
