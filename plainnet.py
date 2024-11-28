from globalvar import *
from tensorflow.keras import layers, models

# initial input is 224*224*3channel RGB
# paper requires: filter doubles when feature map is halved to preserved complexity
def plain_net_cnn_18layer(input_shape: Tuple[int, int, int] = input_size) -> Model:
    model = models.Sequential()
    # conv1: l1
    # (7x7, 64 filters) with stride 2, output 112*112
    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same', strides=(2, 2), input_shape=input_shape))

    # conv2_x:
    # layers l2 - l5, max pooling has no parameters, not a layer
    # Max pooling layer with stride 2, downsample by 2
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # conv3_x: conv3_x (2 x [3x3, 128]) - Downsample by 2
    # layers: l6 - l9
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # Layer 3: conv4_x (2 x [3x3, 256]) - Downsample by 2
    # layers: l10 - l13
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # Layer 4: conv5_x (2 x [3x3, 512]) - No more downsampling, just maintaining size
    # layers: l14 - l17
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # layers: l19, the Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))  # Output layer for 10 classes
    return model

def gen_plainnet_model(
        input_shape: Tuple[int, int, int] = input_size,
        conv2_factor: int = 2, conv3_factor: int = 2,
        conv4_factor: int = 2, conv5_factor: int = 2) -> Model:

    model = models.Sequential()

    # conv1: 1 layer
    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same', strides=(2, 2), input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    total_count = 1

    # conv2_x: conv2_layers
    cur_count = 2*conv2_factor
    total_count += cur_count

    for _ in range(cur_count):
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # conv3_x: 1 layer for downsampling, conv3_layers
    cur_count = 2*conv3_factor
    total_count += cur_count

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2)))
    for _ in range(cur_count-1):
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # conv4_x: 1 layer for downsampling, conv4_layers
    cur_count = 2*conv4_factor
    total_count += cur_count

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2)))
    for _ in range(cur_count-1):
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # conv5_x: conv5_layers (no downsampling)
    cur_count = 2*conv5_factor
    total_count += cur_count
    for _ in range(cur_count):
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)))

    # Flatten and output layer
    total_count += 1
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))  # Output layer for 10 classes
    print(f'this model has layers {total_count}')
    return model


plainnet_18 = gen_plainnet_model()
plainnet_34 = gen_plainnet_model(input_size, 3, 4, 6, 3)
