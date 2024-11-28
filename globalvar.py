import time
import numpy as np
import os
import path
import pydot
from typing import List, Tuple
from matplotlib.pyplot import imshow
# %matplotlib inline
import matplotlib.pyplot as plt
import PIL.Image
import pathlib
import shutil

import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
from preprocessing import image

from keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model

from tensorflow.python.keras.utils import layer_utils
#from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
import scipy.misc

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last') # can be channels_first or channels_last.
# K.set_learning_phase(1) # maki: learning phase is deprecated for new Keras
print("env setup successfully")

# global env

# img_height = 64
# img_width = 64
# batch_size = 32
# image_size = (img_width, img_height)
augmentation_multiplier = 2
patch_size = 224
input_size = [patch_size, patch_size, 3]

# in_folder = os.path.join('..', 'input', 'animals10', 'raw-img')
# middle_folder = os.path.join('..', 'output', 'animals10', 'middle-img')
# out_folder = os.path.join('..', 'output', 'animals10', 'cooked-img')
in_folder = os.path.join('.', 'data', 'raw-img')
middle_folder = os.path.join('.', 'data', 'middle-img')
out_folder = os.path.join('.', 'data', 'cooked-img')
main_folder = os.path.join('.', 'data')
train_folder = out_folder

file_count = []
total = 0
for fld in os.listdir(in_folder):
    crt = os.path.join(in_folder, fld)
    image_count = len(os.listdir(crt))
    file_count.append(image_count)
    total += image_count
    print(f'{crt} contains {image_count} images')

print(f'Raw input {total} num of original data of 10 classes.')

# im_per_class = min(file_count)
# print(f'but we use {im_per_class} for each set to balance input.')

print(f"in_folder is {in_folder}")
print(f"middle_folder is {middle_folder}")
print(f"out_folder is {out_folder}")
