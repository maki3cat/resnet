"""Module of CONFIGURATION"""
import os


# DATA AUGMENTATION
AUGMENT_MULTIPLIER = 4

# TRAINING HyperParameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
TOTAL_CLASS = 10

BATCH_SIZE_18 = 32
BATCH_SIZE_34 = 64

PATCH_SIZE = 224 # used for data augmentation
INPUT_SIZE = (IMG_HEIGHT, IMG_WIDTH, 3)

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
