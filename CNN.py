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
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode="categorical")
    testing_datagen = test_datagen.flow_from_directory(
        testing_dataset,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
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
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    # Training done here
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # CategoricalCrossentropy
                  metrics=['accuracy'])

    print(model.summary())

    epochs = 30
    history = model.fit(
        training_datagen,
        validation_data=testing_datagen,
        epochs=epochs,
        verbose=2
    )
    # dot_img_file = str(Path.cwd() / 'model1.png')
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    #
    # import tensorflow as tf
    # from keras.models import Sequential
    # from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
    # from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
    # img_path = str(Path.cwd() / 'train' / 'colorado hairstreak' / '000001.jpg')  # dog
    # # Define a new Model, Input= image
    # # Output= intermediate representations for all layers in the
    # # previous model after the first.
    # successive_outputs = [layer.output for layer in model.layers[1:]]
    # # visualization_model = Model(img_input, successive_outputs)
    # visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # # Load the input image
    # img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # # Convert ht image to Array of dimension (150,150,3)
    # x = img_to_array(img)
    # x = x.reshape((1,) + x.shape)
    # # Rescale by 1/255
    # x /= 255.0
    # # Let's run input image through our vislauization network
    # # to obtain all intermediate representations for the image.
    # successive_feature_maps = visualization_model.predict(x)
    # # Retrieve are the names of the layers, so can have them as part of our plot
    # layer_names = [layer.name for layer in model.layers]
    # for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    #     print(feature_map.shape)
    #     if len(feature_map.shape) == 4:
    #
    #         # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
    #
    #         n_features = feature_map.shape[-1]  # number of features in the feature map
    #         size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)
    #
    #         # We will tile our images in this matrix
    #         display_grid = np.zeros((size, size * n_features))
    #
    #         # Postprocess the feature to be visually palatable
    #         for i in range(n_features):
    #             x = feature_map[0, :, :, i]
    #             x -= x.mean()
    #             x /= x.std()
    #             x *= 64
    #             x += 128
    #             x = np.clip(x, 0, 255).astype('uint8')
    #             # Tile each filter into a horizontal grid
    #             display_grid[:, i * size: (i + 1) * size] = x
    #         # Display the grid
    #         scale = 20. / n_features
    #         plt.figure(figsize=(scale * n_features, scale))
    #         plt.title(layer_name)
    #         plt.grid(False)
    #         plt.imshow(display_grid, aspect='auto', cmap='viridis')


    # # Iterate thru all the layers of the model
    # for layer in model.layers:
    #     if 'conv' in layer.name:
    #         weights, bias = layer.get_weights()
    #
    #         # normalize filter values between  0 and 1 for visualization
    #         f_min, f_max = weights.min(), weights.max()
    #         filters = (weights - f_min) / (f_max - f_min)
    #         filter_cnt = 1
    #
    #         # plotting all the filters
    #         for i in range(filters.shape[3]):
    #             # get the filters
    #             filt = filters[:, :, :, i]
    #             # plotting each of the channel, color image RGB channels
    #             for j in range(filters.shape[0]):
    #                 ax = plt.subplot(filters.shape[3], filters.shape[0], filter_cnt)
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
    #                 plt.imshow(filt[:, :, j])
    #                 filter_cnt += 1
    #         plt.show()





    # retrieve weights from the second hidden layer
    # filters, biases = model.layers[1].get_weights()
    # # normalize filter values to 0-1 so we can visualize them
    # f_min, f_max = filters.min(), filters.max()
    # filters = (filters - f_min) / (f_max - f_min)
    # # plot first few filters
    # n_filters, ix = 6, 1
    # for i in range(n_filters):
    #     # get the filter
    #     f = filters[:, :, :, i]
    #     # plot each channel separately
    #     for j in range(3):
    #         # specify subplot and turn of axis
    #         ax = plt.subplot(n_filters, 3, ix)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         # plot filter channel in grayscale
    #         plt.imshow(f[:, :, j], cmap='gray')
    #         ix += 1
    # show the figure



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
