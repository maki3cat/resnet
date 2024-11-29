from globalvar import *

def add_conv_block(X, filters, kernel_size, strides=(1,1), layer_num=None):
    conv_name = f'conv_{layer_num}' if layer_num else 'conv'
    bn_name = f'bn_{layer_num}' if layer_num else 'bn'
    X = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding='same',
               kernel_initializer=glorot_uniform(seed=0),
               name=conv_name)(X)
    X = BatchNormalization(axis=3, name=bn_name)(X)
    X = Activation('relu')(X)
    return X

def gen_plainet_model(
        model_name: str = 'plain_net',
        input_shape: tuple[int, int, int] = input_size,  # Assuming this is your input_size
        conv2_factor: int = 2, conv3_factor: int = 2,
        conv4_factor: int = 2, conv5_factor: int = 2,
        total_class: int = 10) -> Model:

    # Input tensor
    inputs = Input(shape=input_shape)
    X = inputs

    # conv1: 1 layer
    X = add_conv_block(X, 64, (7,7), (2,2), layer_num=1)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    total_count = 1

    # conv2_x: conv2_layers
    cur_count = 2*conv2_factor
    total_count += cur_count
    for i in range(cur_count):
        X = add_conv_block(X, 64, (3,3), layer_num=f'2_{i+1}')

    # conv3_x: 1 layer for downsampling, conv3_layers
    cur_count = 2*conv3_factor
    total_count += cur_count
    X = add_conv_block(X, 128, (3,3), (2,2), layer_num='3_1')
    for i in range(cur_count-1):
        X = add_conv_block(X, 128, (3,3), layer_num=f'3_{i+2}')

    # conv4_x: 1 layer for downsampling, conv4_layers
    cur_count = 2*conv4_factor
    total_count += cur_count
    X = add_conv_block(X, 256, (3,3), (2, 2), layer_num='4_1')
    for i in range(cur_count-1):
        X = add_conv_block(X, 256, (3,3), layer_num=f'4_{i+2}')

    # conv5_x: conv5_layers (no downsampling)
    cur_count = 2*conv5_factor
    total_count += cur_count
    for i in range(cur_count):
        X = add_conv_block(X, 512, (3,3), layer_num=f'5_{i+1}')

    # Global Average Pooling layer
    X = GlobalAveragePooling2D()(X)
    # Output layer
    outputs = Dense(total_class, activation='softmax')(X)
    total_count += 1

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    print(f'The model {model_name} has {total_count} layers')
    return model

model_plainet_18 = gen_plainet_model('plain_net_18')
model_plainet_34 = gen_plainet_model('plain_net_34',input_size, 3, 4, 6, 3)
