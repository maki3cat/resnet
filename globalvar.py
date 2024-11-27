import os

# global env regarding data preprocessing
in_folder = os.path.join('.', 'data', 'raw-img')
out_folder = os.path.join('.', 'data', 'cooked-img')
img_height = 64
img_width = 64
batch_size = 32
image_size = (img_width, img_height)
