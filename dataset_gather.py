import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from icrawler.builtin import BingImageCrawler

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