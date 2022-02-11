import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from PIL import Image
from pathlib2 import Path
from icrawler.builtin import BingImageCrawler
from sklearn.model_selection import train_test_split

def download_pictures(name_of_butterfly,num_of_pics):
    bing_crawler = BingImageCrawler(downloader_threads=2,storage={'root_dir': name_of_butterfly})
    bing_crawler.crawl(keyword=name_of_butterfly, offset=0, max_num=num_of_pics)
    print(f"Printed {name_of_butterfly} pictures successfully")


def keep_delete(input,filename):
    if input == 'y':
        pass
    if input == 'n':
        os.remove(filename)


def verify_pictures(main_dir):
    for (dirpath, dirnames, filenames) in os.walk(main_dir):
        for name in filenames:
            img = os.path.join(dirpath, name)
            image = mpimg.imread(img)
            imgplot = plt.imshow(image)
            plt.pause(0.5)
            keep_delete_input = input("Press y if image is good and n if image is bad")
            keep_delete(keep_delete_input,img)
            plt.close()


def split_train_test(main_dir):
    """
    Go to butterfly_picture
    go to each subdirectory
    get name of each
    then len of sub directory (want 80 percent of the pictures for train 20 percent for test)
    :return:
    """
    current_dir = str(Path.cwd())
    for (dirpath, dirnames, filenames) in os.walk(main_dir): # Iterates through only once
        for subdir in dirnames:
            train_path = current_dir + '/train/' + subdir + '/'
            test_path = current_dir + '/test/' + subdir + '/'
            make_dir(train_path)
            make_dir(test_path)
            for (_,_,files) in os.walk(dirpath + subdir):
                x_train, x_test = train_test_split(files, test_size=0.2)
                for file in x_train:
                    Path(dirpath + subdir + '/' + file).rename(train_path + file)
                for file in x_test:
                    Path(dirpath + subdir + '/' + file).rename(test_path + file)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)