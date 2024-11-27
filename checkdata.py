import os
import pathlib
import PIL.Image
# path to desired image set, relative to current working dir
import matplotlib.pyplot as plt
from globalvar import in_folder, out_folder

def check_raw_data():
    file_count = []
    # get number of images in each folder (images per class)
    for fld in os.listdir(in_folder):
        crt = os.path.join(in_folder, fld)
        image_count = len(os.listdir(crt))
        file_count.append(image_count)
        print(f'{crt} contains {image_count} images')

    os.listdir(os.path.join(in_folder, 'elefante'))[:10]
    data_dir = pathlib.Path(in_folder)
    elefante = list(data_dir.glob('elefante/*'))
    im = PIL.Image.open(str(elefante[0]))
    print(PIL.Image.open(str(elefante[0])).size)
    print(PIL.Image.open(str(elefante[10])).size)
    height, width, channels = im.shape
    print(f"Image shape: {image.shape}")
    print(f"Height: {height}, Width: {width}, Channels: {channels}")
    plt.imshow(im)
    plt.show()


def check_cooked_data():
    data_dir = pathlib.Path(out_folder)
    elefante = list(data_dir.glob('elefante/*'))
    im = PIL.Image.open(str(elefante[0]))
    # Get the shape
    width, height = im.size
    print(f"Image size: {im.size}")
    print(f"Width: {width}, Height: {height}")
    plt.imshow(im)
    plt.show()

    data_dir = pathlib.Path(out_folder)
    mucca = list(data_dir.glob('mucca/*'))
    im = PIL.Image.open(str(mucca[5]))
    width, height = im.size
    print(f"Image size: {im.size}")
    print(f"Width: {width}, Height: {height}")
    plt.imshow(im)
    plt.show()


# check_raw_data()
check_cooked_data()

