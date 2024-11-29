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
import random

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

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

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from datetime import datetime
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last') # can be channels_first or channels_last.
# K.set_learning_phase(1) # maki: learning phase is deprecated for new Keras
print("env setup successfully")

# data augmentation
augmentation_multiplier = 4

# global env
img_height = 128
img_width = 128
image_size = (img_width, img_height)
total_class = 10

batch_size_18 = 32
batch_size_34 = 64

patch_size = 224 # used for data augmentation
input_size = (img_height, img_width, 3)

# in_folder = os.path.join('..', 'input', 'animals10', 'raw-img')
# middle_folder = os.path.join('..', 'output', 'animals10', 'middle-img')
# out_folder = os.path.join('..', 'output', 'animals10', 'cooked-img')
in_folder = os.path.join('.', 'data', 'raw-img')
resized_data_folder = os.path.join('.', 'data', 'resized-img')
out_folder = os.path.join('.', 'data', 'cooked-img')

main_folder = os.path.join('.', 'data')
train_folder = out_folder

file_count = []
print(f"in_folder is {in_folder}")
print(f"resized_folder is {resized_data_folder}")
print(f"out_folder is {out_folder}")
print(f"train_folder is {train_folder}")

# total = 0
# for fld in os.listdir(train_folder):
#     crt = os.path.join(in_folder, fld)
#     image_count = len(os.listdir(crt))
#     file_count.append(image_count)
#     total += image_count
#     print(f'{crt} contains {image_count} images')
# print(f'Total data for training and validatation is {total} num of {total_class} classes.')
