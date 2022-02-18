import matplotlib
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from pathlib2 import Path
from Constants import IMG_HEIGHT, IMG_WIDTH, ITERATIONS

matplotlib.use('TkAgg')


def plot_accuracy(acc, val_acc, loss, val_loss, epochs_range):
    if multi_accuracy:
        fig, ax = plt.subplots(2, figsize=(13, 8))
        for i in range(0, ITERATIONS):
            ax[0].plot(epochs_range, acc[i], label=f'Training Acc Iteration {i}')
            ax[0].plot(epochs_range, val_acc[i], label=f'Validation Acc Iteration {i}')
            ax[0].set_ylabel("Accuracy Percentage")
            ax[0].legend(loc='lower right')
            ax[0].set_title('Training and Validation Accuracy')
            ax[0].grid()

            ax[1].plot(epochs_range, loss[i], label=f'Training Loss Iteration {i}')
            ax[1].plot(epochs_range, val_loss[i], label=f'Validation Loss Iteration {i}')
            ax[1].set_xlabel("Number of Epochs")
            ax[1].legend(loc='upper right')
            ax[1].set_title('Training and Validation Loss')
            ax[1].grid()
    else:
        poly_train_acc_x, poly_train_acc_y = _calc_polyfit(epochs_range, acc)
        poly_val_acc_x, poly_val_acc_y = _calc_polyfit(epochs_range, val_acc)

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

        poly_train_loss_x, poly_train_loss_y = _calc_polyfit(epochs_range, loss)
        poly_val_loss_x, poly_val_loss_y = _calc_polyfit(epochs_range, val_loss)

        ax[1, 0].plot(epochs_range, loss, label='Training Loss')
        ax[1, 0].plot(epochs_range, val_loss, label='Validation Loss')
        ax[1, 0].set_xlabel("Number of Epochs")
        ax[1, 0].legend(loc='upper right')
        ax[1, 0].set_title('Training and Validation Loss')
        ax[1, 0].grid()

        ax[1, 1].plot(poly_train_loss_x, poly_train_loss_y, '-', label='Training Loss')
        ax[1, 1].plot(poly_val_loss_x, poly_val_loss_y, '--', label='Validation Loss')
        ax[1, 1].set_xlabel("Number of Epochs")
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set_title('Training and Validation Loss Regression Lines')
        ax[1, 1].grid()

def plot_multi_accuracy(acc, val_acc, loss, val_loss, epochs_range):
    fig, ax = plt.subplots(2, figsize=(13, 8))
    for i in range(0, ITERATIONS):
        ax[0].plot(epochs_range, acc[i], label=f'Training Acc Iteration {i}')
        ax[0].plot(epochs_range, val_acc[i], label=f'Validation Acc Iteration {i}')
        ax[0].set_ylabel("Accuracy Percentage")
        ax[0].legend(loc='lower right')
        ax[0].set_title('Training and Validation Accuracy')
        ax[0].grid()

        ax[1].plot(epochs_range, loss[i], label=f'Training Loss Iteration {i}')
        ax[1].plot(epochs_range, val_loss[i], label=f'Validation Loss Iteration {i}')
        ax[1].set_xlabel("Number of Epochs")
        ax[1].legend(loc='upper right')
        ax[1].set_title('Training and Validation Loss')
        ax[1].grid()

def plot_model_outline(model):
    dot_img_file = str(Path.cwd() / 'cnn_outline_model.png')
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


def plot_layers(model):
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
    img_path = str(Path.cwd() / 'train' / 'colorado hairstreak' / '000001.jpg')  # dog
    # Define a new Model, Input= image
    # Output= intermediate representations for all layers in the
    # previous model after the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # Load the input image
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Convert ht image to Array of dimension (150,150,3)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # Rescale by 1/255
    x /= 255.0
    # Let's run input image through our vislauization network
    # to obtain all intermediate representations for the image.
    successive_feature_maps = visualization_model.predict(x)
    # Retrieve are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        print(feature_map.shape)
        if len(feature_map.shape) == 4:

            # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers

            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # Postprocess the feature to be visually palatable
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # Tile each filter into a horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')


def plot_feature_map(model):
    from keras.preprocessing import image
    import numpy as np
    img_path = str(Path.cwd() / 'train' / 'colorado hairstreak' / '000001.jpg')
    # Pre-processing the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.

    # Print image tensor shape
    print(img_tensor.shape)

    from keras import models
    plt.imshow(img_tensor[0])
    plt.show()

    layer_outputs = [layer.output for layer in model.layers[:6]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    """
    Output model layers but only the layers of conv2d
    Next based off the shape get the first and last index
    Use this as the plt.matshow() 4 parameter
    """

    # num_of_activations = model / 2

    # Getting Activations of first layer
    first_layer_activation = activations[2]

    # shape of first layer activation
    print(first_layer_activation.shape)

    # 6th channel of the image after first layer of convolution is applied

    plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')

    # 15th channel of the image after first layer of convolution is applied
    plt.matshow(first_layer_activation[0, :, :, 15], cmap='viridis')


def _calc_polyfit(num_of_epochs, accuracy):
    coefficients = np.polyfit(num_of_epochs, accuracy, 3)
    poly = np.poly1d(coefficients)
    new_x = np.linspace(num_of_epochs[0], num_of_epochs[-1])
    new_y = poly(new_x)
    return new_x, new_y
