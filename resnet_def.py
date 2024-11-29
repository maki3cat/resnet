from globalvar import *

def dimension_match_block(X: tf.Tensor, level: int, block: int, f: int, s: tuple[int,int]=(2, 2)) -> tf.Tensor:
    conv_name = f'conv{level}_{block}' + '_{layer}_{type}'
    f1, f2, f3 = f, f, f
    X_shortcut = X
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=s, padding='valid',
               name=conv_name.format(layer=1, type='conv'),
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
    X = Activation('relu', name=conv_name.format(layer=1, type='relu'))(X)

    X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=s, padding='valid',
                        name=conv_name.format(layer='short', type='conv'),
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=conv_name.format(layer='short', type='bn'))(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu', name=conv_name.format(layer=3, type='relu'))(X)
    return X

def dimension_match_block(X, level, block, f, s=(2, 2)):
    conv_name_base = f'conv{level}_{block}_branch'
    bn_name_base = f'bn{level}_{block}_branch'
    X_shortcut = X

    X = Conv2D(f, (1, 1), strides=s, name=f'{conv_name_base}2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'{bn_name_base}2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f, (3, 3), padding='same', name=f'{conv_name_base}2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'{bn_name_base}2b')(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(f, (1, 1), strides=s, name=f'{conv_name_base}1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=f'{bn_name_base}1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def identity_block(X: tf.Tensor, level: int, block: int, f: int) -> tf.Tensor:
    conv_name_base = f'conv{level}_{block}_branch'
    bn_name_base = f'bn{level}_{block}_branch'
    X_shortcut = X

    X = Conv2D(f, (3, 3), padding='same', name=f'{conv_name_base}2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'{bn_name_base}2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f, (3, 3), padding='same', name=f'{conv_name_base}2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'{bn_name_base}2b')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def gen_resnet_18(input_size: tuple[int,int,int]=input_size, classes: int=total_class) -> Model:
    X_input = Input(input_size)
    X = ZeroPadding2D((3, 3))(X_input)
    # conv1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
               name='conv1_1_1_conv',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='conv1_1_1_nb')(X)
    X = Activation('relu')(X)

    # conv2_x
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = dimension_match_block(X, level=2, block=1, f=64, s=(1, 1))
    X = identity_block(X, level=2, block=1, f=64)

    # conv3_x
    X = dimension_match_block(X, level=2, block=1, f=128, s=(2, 2))
    X = identity_block(X, level=2, block=1, f=128)

    # conv4_x
    X = dimension_match_block(X, level=2, block=1, f=256, s=(2, 2))
    X = identity_block(X, level=2, block=1, f=256)

    # conv5_x
    X = dimension_match_block(X, level=2, block=1, f=512, s=(2, 2))
    X = identity_block(X, level=2, block=1, f=512)

    # Pooling blocks
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc_' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input, outputs=X, name='res_net_18')
    return model

model_resnet_18 = gen_resnet_18()
model_resnet_34 = gen_resnet_34()
