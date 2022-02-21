"""
Image Classification
Description of File: File where manipulating dataset to from test and training data is done
Author: John Halter
Last Updated: 02/21/22
"""
import os

from pathlib2 import Path
from sklearn.model_selection import train_test_split

from dataset_creation.dataset_gather import make_dir


def split_train_test(main_dir):
    """
    Function where each dataset is split into training and testing directories for model learning
    :param main_dir: the main_directory directory where images are currently held
    :return: places images into new directory for learning
    """
    current_dir = str(Path.cwd())
    for (dirpath, dirnames, filenames) in os.walk(main_dir):  # Iterates through only once
        for subdir in dirnames:
            train_path = current_dir + '/train/' + subdir + '/'
            test_path = current_dir + '/test/' + subdir + '/'
            make_dir(train_path)
            make_dir(test_path)
            for (_, _, files) in os.walk(dirpath + subdir):
                x_train, x_test = train_test_split(files, test_size=0.2)
                for file in x_train:
                    Path(dirpath + subdir + '/' + file).rename(train_path + file)
                for file in x_test:
                    Path(dirpath + subdir + '/' + file).rename(test_path + file)