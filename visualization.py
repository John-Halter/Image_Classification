"""
Image Classification
Description of File: File where visualization function are held
Author: John Halter
Last Updated: 02/21/22
"""
import os

import matplotlib
import tensorflow as tf
import numpy as np

from keras.preprocessing import image
from keras import models
from matplotlib import pyplot as plt
from pathlib2 import Path
from Constants import IMG_HEIGHT, IMG_WIDTH
from calculated_values import calc_polyfit
from dataset_creation.dataset_gather import make_dir

matplotlib.use('TkAgg')


def plot_accuracy(acc, val_acc, loss, val_loss, epochs_range):
    """
    Function to plot the accuracy of the cnn model
    :param acc: A list with the accuracy of the training dataset
    :param val_acc: A list with the accuracy of the validation dataset
    :param loss: A list with the loss of the training dataset
    :param val_loss: A list with the loss of the validation dataset
    :param epochs_range: The number of epochs
    :return: plot of the accuracy and loss of the cnn model
    """
    poly_train_acc_x, poly_train_acc_y = calc_polyfit(epochs_range, acc)
    poly_val_acc_x, poly_val_acc_y = calc_polyfit(epochs_range, val_acc)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(epochs_range, acc, label='Training Accuracy')
    ax[0, 0].plot(epochs_range, val_acc, label='Validation Accuracy')
    ax[0, 0].set_ylabel("Accuracy Percentage")
    ax[0, 0].legend(loc='lower right')
    ax[0, 0].set_title('Training and Validation Accuracy')
    ax[0, 0].grid()

    ax[0, 1].plot(poly_train_acc_x, poly_train_acc_y, '-', label='Regression Training Acc')
    ax[0, 1].plot(poly_val_acc_x, poly_val_acc_y, '--', label='Regression Testing Acc')
    ax[0, 1].set_ylabel("Accuracy Percentage")
    ax[0, 1].legend(loc='lower right')
    ax[0, 1].set_title('Training and Validation Accuracy Regression Lines')
    ax[0, 1].grid()

    poly_train_loss_x, poly_train_loss_y = calc_polyfit(epochs_range, loss)
    poly_val_loss_x, poly_val_loss_y = calc_polyfit(epochs_range, val_loss)

    ax[1, 0].plot(epochs_range, loss, label=' Regression Training Loss')
    ax[1, 0].plot(epochs_range, val_loss, label='Regression Validation Loss')
    ax[1, 0].set_xlabel("Number of Epochs")
    ax[1, 0].set_ylabel("Loss Value")
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].set_title('Training and Validation Loss Regression Lines')
    ax[1, 0].grid()

    ax[1, 1].plot(poly_train_loss_x, poly_train_loss_y, '-', label='Training Loss')
    ax[1, 1].plot(poly_val_loss_x, poly_val_loss_y, '--', label='Validation Loss')
    ax[1, 1].set_xlabel("Number of Epochs")
    ax[1, 1].set_ylabel("Loss Value")
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].set_title('Training and Validation Loss Regression Lines')
    ax[1, 1].grid()

    accuracy_plot_path = str(Path.cwd() / 'images')
    make_dir(accuracy_plot_path)
    fig.savefig(accuracy_plot_path + '/model_accuracy_loss_plot.png')

def plot_model_outline(model):
    """
    Function to plot and save cnn model structure
    :param model: The cnn model
    :return: saves the image to local file directories
    """
    model_img_dir = str(Path.cwd() / 'images')
    make_dir(model_img_dir)
    model_img_path = model_img_dir + '/cnn_outline_model.png'
    tf.keras.utils.plot_model(model, to_file=model_img_path, show_shapes=True)


def plot_example_feature_map(model):
    """
    Function to plot image from dataset and the feature map for that specified image
    :param model: The cnn model
    :return: Two plots of the image and the feature map from the cnn model
    """
    img_path = str(Path.cwd() / 'train' / 'colorado hairstreak' / '000001.jpg')
    # Pre-processing the image
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.
    # Plotting image used
    fig, ax = plt.subplots()
    ax.imshow(img_tensor[0])
    # Getting the layer of the model to use for the feature map
    layer_outputs = [layer.output for layer in model.layers[:6]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs[1])
    activations = activation_model.predict(img_tensor)
    # Plotting feature map
    plt.figure(figsize=(10, 10))
    for i in range(1, activations.shape[3] + 1):
        plt.subplot(8, 8, i)
        plt.imshow(activations[0, :, :, i - 1], cmap='viridis')
        plt.axis('off')
    plt.show()