"""
Image Classification
Description of File: File where calculations are stored
Author: John Halter
Last Updated: 02/21/22
"""
import numpy as np


def calc_polyfit(num_of_epochs, accuracy):
    """
    Function to calculate regression line
    :param num_of_epochs: length of x axis to use
    :param accuracy: The accuracy of the cnn model
    :return: Two new values used to plot regression line
    """
    coefficients = np.polyfit(num_of_epochs, accuracy, 3)
    poly = np.poly1d(coefficients)
    new_x = np.linspace(num_of_epochs[0], num_of_epochs[-1])
    new_y = poly(new_x)
    return new_x, new_y
