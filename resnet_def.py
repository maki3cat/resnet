from globalvar import *

def dimension_match_block(X: tf.Tensor, f: int, s: tuple[int,int]=(2, 2)) -> tf.Tensor:
    X_shortcut = X
    # First convolution
    X = Conv2D(filters=f, kernel_size=(1, 1), strides=s, padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second convolution
    X = Conv2D(filters=f, kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = Conv2D(filters=f, kernel_size=(1, 1), strides=s,
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # Add shortcut to main path and apply activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def identity_block(X: tf.Tensor, f: int) -> tf.Tensor:
    X_shortcut = X
    # First convolution
    X = Conv2D(filters=f, kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second convolution
    X = Conv2D(filters=f, kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    # Add shortcut to main path and apply activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def gen_resnet_18(input_size: tuple[int,int,int]=input_size, classes: int=total_class) -> Model:
    X_input = Input(shape=input_size)
    # Initial padding and conv1
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Max pooling
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # conv2_x
    X = dimension_match_block(X, f=64)
    X = identity_block(X, f=64)

    # conv3_x
    X = dimension_match_block(X, f=128)
    X = identity_block(X, f=128)

    # conv4_x
    X = dimension_match_block(X, f=256)
    X = identity_block(X, f=256)

    # conv5_x
    X = dimension_match_block(X, f=512)
    X = identity_block(X, f=512)

    # Pooling and output layer
    X = GlobalAveragePooling2D()(X)
    # Flatten and dense layer for classification
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X)
    return model

def gen_resnet_34(input_size: tuple[int,int,int]=input_size, classes: int=total_class) -> Model:
    X_input = Input(shape=input_size)
    # Initial padding and conv1
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Max pooling
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # conv2_x
    X = dimension_match_block(X, f=64)
    X = identity_block(X, f=64)
    X = identity_block(X, f=64)

    # conv3_x
    X = dimension_match_block(X, f=128)
    X = identity_block(X, f=128)
    X = identity_block(X, f=128)
    X = identity_block(X, f=128)

    # conv4_x
    X = dimension_match_block(X, f=256)
    X = identity_block(X, f=256)
    X = identity_block(X, f=256)
    X = identity_block(X, f=256)
    X = identity_block(X, f=256)
    X = identity_block(X, f=256)

    # conv5_x
    X = dimension_match_block(X, f=512)
    X = identity_block(X, f=512)
    X = identity_block(X, f=512)

    # Pooling and output layer
    # TODO: what is the difference btw the 2 pooling?
    X = GlobalAveragePooling2D()(X)
    # X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # Flatten and dense layer for classification
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X)
    return model

model_resnet_18 = gen_resnet_18()
model_resnet_34 = gen_resnet_34()
