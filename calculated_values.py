import numpy as np


def calc_polyfit(num_of_epochs, accuracy):
    coefficients = np.polyfit(num_of_epochs, accuracy, 3)
    poly = np.poly1d(coefficients)
    new_x = np.linspace(num_of_epochs[0], num_of_epochs[-1])
    new_y = poly(new_x)
    return new_x, new_y
