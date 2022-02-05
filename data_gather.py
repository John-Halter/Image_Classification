from bing_image_downloader import downloader

def download_pictures(ls_of_names,num_of_pics, output_directory):
    for i in ls_of_names:
        downloader.download(query=f"{i} Butterfly",limit=num_of_pics,force_replace=True,output_dir=output_directory)

    print("Printed All pictures successfully")