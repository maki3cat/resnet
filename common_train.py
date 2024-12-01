"""Module for common training functions and dataset preparation."""

import pathlib
import numpy as np
import keras
import tensorflow as tf
from config import (IMG_HEIGHT, IMG_WIDTH, train_folder)
class_names = []
def split_ds_training(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split the dataset into normalized training and validation sets.

    This function loads images from a directory, splits them into training and validation sets,
    applies normalization, and returns the processed datasets.

    Args:
        batch_size (int): The number of samples per batch to load.
    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: A tuple containing the normalized training
        and validation datasets.
    Global Variables:      
        This function assumes the existence of global variables train_folder, IMG_HEIGHT,
        IMG_WIDTH, and class_names. It modifies class_names by extending it with the
        class names found in the dataset. 
   """
    data_dir = pathlib.Path(train_folder)
    # training and validation
    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        label_mode='categorical',
        seed=100,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        label_mode='categorical',
        seed=100,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size
    )
    class_names.extend(train_ds.class_names)
    # Create normalization layer
    normalization_layer = keras.Sequential([
        keras.layers.Rescaling(1./255)
    ])

    # Normalize datasets
    norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    image_batch, _ = next(iter(norm_train_ds))
    first_image = image_batch[0]
    print("Pixel value range:", np.min(first_image), "-", np.max(first_image))
    return norm_train_ds, norm_val_ds

training_callback = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=10,
        verbose=1
    )
]
print("Class names:", class_names)
