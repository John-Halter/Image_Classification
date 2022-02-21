"""
Image Classification
Description of File: The main file where you can access all functions by setting different conditions to true
Author: John Halter
Last Updated: 02/21/22
"""
from matplotlib import pyplot as plt
from pathlib2 import Path

from dataset_creation.dataset_gather import download_pictures, verify_pictures
from dataset_creation.dataset_manipulation import split_train_test
from Constants import LIST_OF_SPECIES, OUTPUT_DIR, NUM_OF_PICS
from cnn import cnn_model
from visualization import plot_accuracy, plot_model_outline, plot_example_feature_map

if __name__ == '__main__':
    # Training and testing dataset paths
    training_dataset = Path.cwd() / 'train'
    testing_dataset = Path.cwd() / 'test'

    # Conditions to access functionality of code
    download_pics = False
    # Verify if pictures are worth keeping
    verify_pics = False
    # Split the dataset into the different sections
    split = False
    # Run the cnn model
    run_cnn = True
    # Print the model structure
    model_outline = True
    # Print an example of the features of the model for an image
    feature = False

    if download_pics:
        for i in range(len(LIST_OF_SPECIES)):
            download_pictures(LIST_OF_SPECIES[i], NUM_OF_PICS)

    if verify_pics:
        verify_pictures(OUTPUT_DIR)

    if split:
        split_train_test(OUTPUT_DIR)

    if run_cnn:
        model, acc, val_acc, loss, val_loss, epochs_range = cnn_model(training_dataset, testing_dataset)
        plot_accuracy(acc, val_acc, loss, val_loss, epochs_range)
        if model_outline:
            plot_model_outline(model)
        if feature:
            plot_example_feature_map(model)

plt.show()
