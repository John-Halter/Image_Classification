"""
Image Classification
Description of File: File where gathering and stored dataset is done
Author: John Halter
Last Updated: 02/21/22
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from icrawler.builtin import BingImageCrawler


def download_images(name_of_labels, num_of_pics):
    """
    Function to download images and put into seperate directories based on label
    :param name_of_labels: The name of the labels for classification
    :param num_of_pics: number of pictures to download for each label
    :return:
    """
    bing_crawler = BingImageCrawler(downloader_threads=2, storage={'root_dir': name_of_labels})
    bing_crawler.crawl(keyword=name_of_labels, offset=0, max_num=num_of_pics)
    print(f"Printed {name_of_labels} pictures successfully")


def _keep_delete(input, filename):
    """
    Local function to either keep and image or remove function from directory
    :param input: User keyboard input
    :param filename: File to be removed
    :return:
    """
    if input.lower() == 'y':
        pass
    if input.lower == 'n':
        os.remove(filename)


def verify_images(main_dir):
    """
    Function to iterate through directory of images and verify if they are to be kept or removed
    :param main_dir: The directory
    :return:
    """
    for (dirpath, dirnames, filenames) in os.walk(main_dir):
        for name in filenames:
            img = os.path.join(dirpath, name)
            image = mpimg.imread(img)
            imgplot = plt.imshow(image)
            plt.pause(0.5)
            keep_delete_input = input("Press y if image is good and n if image is bad")
            _keep_delete(keep_delete_input, img)
            plt.close()


def make_dir(path):
    """
    Function that makes a directory if it does not exist
    :param path: The path of the directory
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
