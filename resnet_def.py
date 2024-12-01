"""Module for defining ResNet blocks and models."""

import tensorflow as tf
from keras.layers import (
    Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
    Flatten, Conv2D, GlobalAveragePooling2D, MaxPooling2D
)
from keras.initializers import glorot_uniform
from keras.models import Model

from config import INPUT_SIZE, TOTAL_CLASS

def identity_block(model: tf.Tensor, f: int) -> tf.Tensor:
    """
    Create an identity block for ResNet.
    Args:
        model (tf.Tensor): Input tensor.
        f (int): Number of filters.
    Returns:
        tf.Tensor: Output tensor after applying the identity block.
    """
    model_shortcut = model
    # First convolution
    model = Conv2D(filters=f, kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)

    # Second convolution
    model = Conv2D(filters=f, kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(model)
    model = BatchNormalization(axis=3)(model)

    # Add shortcut to main path and apply activation
    model = Add()([model, model_shortcut])
    model = Activation('relu')(model)
    return model

def projection_block(model: tf.Tensor, f: int, s: tuple[int,int]=(2, 2)) -> tf.Tensor:
    """
    Create the projection identity matching block for ResNet.
    Args:
        model (tf.Tensor): Input tensor.
        f (int): Number of filters.
        s (tuple[int,int]): Stride for the first convolution. Defaults to (2, 2).
    Returns:
        tf.Tensor: Output tensor after applying the dimension matching block.
    """
    model_shortcut = model
    # First convolution
    model = Conv2D(filters=f, kernel_size=(1, 1), strides=s, padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)

    # Second convolution
    model = Conv2D(filters=f, kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(model)
    model = BatchNormalization(axis=3)(model)

    # Shortcut path
    model_shortcut = Conv2D(filters=f, kernel_size=(1, 1), strides=s,
                        kernel_initializer=glorot_uniform(seed=0))(model_shortcut)
    model_shortcut = BatchNormalization(axis=3)(model_shortcut)

    # Add shortcut to main path and apply activation
    model = Add()([model, model_shortcut])
    model = Activation('relu')(model)
    return model


def gen_resnet_18(input_shape: tuple[int,int,int]=INPUT_SIZE, classes: int=TOTAL_CLASS) -> Model:
    """
    Generate a ResNet-18 model.
    Args:
        input_shape (tuple[int,int,int]): Shape of the input tensor. Defaults to input_size.
        num_classes (int): Number of output classes. Defaults to total_class.
    Returns:
        Model: The constructed ResNet-18 model.
    """
    model_input = Input(shape=input_shape)
    # Initial padding and conv1
    model = ZeroPadding2D((3, 3))(model_input)
    model = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
               kernel_initializer=glorot_uniform(seed=0))(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)

    # Max pooling
    model = MaxPooling2D((3, 3), strides=(2, 2))(model)

    # conv2_x
    model = projection_block(model, f=64)
    model = identity_block(model, f=64)

    # conv3_x
    model = projection_block(model, f=128)
    model = identity_block(model, f=128)

    # conv4_x
    model = projection_block(model, f=256)
    model = identity_block(model, f=256)

    # conv5_x
    model = projection_block(model, f=512)
    model = identity_block(model, f=512)

    # Pooling and output layer
    model = GlobalAveragePooling2D()(model)
    # Flatten and dense layer for classification
    model = Flatten()(model)
    model = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(model)

    model = Model(inputs=model_input, outputs=model)
    return model

def gen_resnet_34(input_shape: tuple[int,int,int]=INPUT_SIZE, classes: int=TOTAL_CLASS) -> Model:
    """
    Generate a ResNet-34 model.
    Args:
        input_shape (tuple[int,int,int]): Shape of the input tensor. Defaults to input_size.
        num_classes (int): Number of output classes. Defaults to total_class.
    Returns:
        Model: The constructed ResNet-18 model.
    """
    model_input = Input(shape=input_shape)
    # Initial padding and conv1
    model = ZeroPadding2D((3, 3))(model_input)
    model = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
               kernel_initializer=glorot_uniform(seed=0))(model)
    model = BatchNormalization(axis=3)(model)
    model = Activation('relu')(model)

    # Max pooling
    model = MaxPooling2D((3, 3), strides=(2, 2))(model)

    # conv2_x
    model = projection_block(model, f=64)
    model = identity_block(model, f=64)
    model = identity_block(model, f=64)

    # conv3_x
    model = projection_block(model, f=128)
    model = identity_block(model, f=128)
    model = identity_block(model, f=128)
    model = identity_block(model, f=128)

    # conv4_x
    model = projection_block(model, f=256)
    model = identity_block(model, f=256)
    model = identity_block(model, f=256)
    model = identity_block(model, f=256)
    model = identity_block(model, f=256)
    model = identity_block(model, f=256)

    # conv5_x
    model = projection_block(model, f=512)
    model = identity_block(model, f=512)
    model = identity_block(model, f=512)

    # Pooling and output layer
    model = GlobalAveragePooling2D()(model)
    # model = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(model)
    # Flatten and dense layer for classification
    model = Flatten()(model)
    model = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(model)
    model = Model(inputs=model_input, outputs=model)
    return model

model_resnet_18 = gen_resnet_18()
model_resnet_34 = gen_resnet_34()
