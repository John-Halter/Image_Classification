from bing_image_downloader import downloader
from Constants import LIST_OF_SPECIES,output_dir


def download_pictures(ls_of_names,output_directory):
    for i in ls_of_names:
        downloader.download(query=f"{i} Butterfly",limit=100,output_dir=output_directory)

    print("Printed All pictures successfully")


download_pictures(LIST_OF_SPECIES,output_dir)