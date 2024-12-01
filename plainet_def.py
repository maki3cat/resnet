"""Module for defining Plain CNN models and related functions."""

from keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense
)
from keras.initializers import glorot_uniform
from keras.models import Model

from config import INPUT_SIZE

def add_conv_block(model, filters, kernel_size, strides=(1,1), layer_num=None):
    """
    Add a convolutional block to the model.

    Args:
        x (Tensor): Input tensor.
        filters (int): Number of filters in the convolutional layer.
        kernel_size (tuple): Size of the convolutional kernel.
        strides (tuple): Strides of the convolution. Defaults to (1,1).
        layer_num (int, optional): Layer number for naming. Defaults to None.

    Returns:
        Tensor: Output tensor after applying the convolutional block.
    """
    conv_name = f'conv_{layer_num}' if layer_num else 'conv'
    bn_name = f'bn_{layer_num}' if layer_num else 'bn'
    model = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding='same',
               kernel_initializer=glorot_uniform(seed=0),
               name=conv_name)(model)
    model = BatchNormalization(axis=3, name=bn_name)(model)
    model = Activation('relu')(model)
    return model

def gen_plainet_model(
        model_name: str = 'plain_net',
        input_shape: tuple[int, int, int] = INPUT_SIZE,  # Assuming this is your input_size
        conv2_factor: int = 2, conv3_factor: int = 2,
        conv4_factor: int = 2, conv5_factor: int = 2,
        total_class: int = 10) -> Model:
    """
    Generate a PlainNet model.

    Args:
        model_name (str): Name of the model. Defaults to 'plain_net'.
        input_shape (tuple): Shape of the input tensor. Defaults to input_size.
        conv2_factor (int): Factor for conv2 layers. Defaults to 2.
        conv3_factor (int): Factor for conv3 layers. Defaults to 2.
        conv4_factor (int): Factor for conv4 layers. Defaults to 2.
        conv5_factor (int): Factor for conv5 layers. Defaults to 2.
        num_classes (int): Number of output classes. Defaults to 10.

    Returns:
        Model: The constructed PlainNet model.
    """
    # Input tensor
    inputs = Input(shape=input_shape)
    model = inputs

    # conv1: 1 layer
    model = add_conv_block(model, 64, (7,7), (2,2), layer_num=1)
    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)
    total_count = 1

    # conv2_x: conv2_layers
    cur_count = 2*conv2_factor
    total_count += cur_count
    for i in range(cur_count):
        model = add_conv_block(model, 64, (3,3), layer_num=f'2_{i+1}')

    # conv3_x: 1 layer for downsampling, conv3_layers
    cur_count = 2*conv3_factor
    total_count += cur_count
    model = add_conv_block(model, 128, (3,3), (2,2), layer_num='3_1')
    for i in range(cur_count-1):
        model = add_conv_block(model, 128, (3,3), layer_num=f'3_{i+2}')

    # conv4_x: 1 layer for downsampling, conv4_layers
    cur_count = 2*conv4_factor
    total_count += cur_count
    model = add_conv_block(model, 256, (3,3), (2, 2), layer_num='4_1')
    for i in range(cur_count-1):
        model = add_conv_block(model, 256, (3,3), layer_num=f'4_{i+2}')

    # conv5_x: conv5_layers (no downsampling)
    cur_count = 2*conv5_factor
    total_count += cur_count
    for i in range(cur_count):
        model = add_conv_block(model, 512, (3,3), layer_num=f'5_{i+1}')

    # Global Average Pooling layer
    model = GlobalAveragePooling2D()(model)
    # Output layer
    outputs = Dense(total_class, activation='softmax')(model)
    total_count += 1

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    print(f'The model {model_name} has {total_count} layers')
    return model

model_plainet_18 = gen_plainet_model('plain_net_18')
model_plainet_34 = gen_plainet_model('plain_net_34',INPUT_SIZE, 3, 4, 6, 3)
