import os
import pathlib
import PIL.Image
# path to desired image set, relative to current working dir
import matplotlib.pyplot as plt

def check_data():
    file_count = []
    in_folder = os.path.join('.', 'data', 'raw-img')
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
    plt.imshow(im)
    plt.show()

check_data()
