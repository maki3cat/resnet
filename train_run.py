"""This module is the training script on top of ready to train data which takes long and GPU to run"""

import time
from datetime import datetime

import tensorflow as tf

from config import BATCH_SIZE_18, BATCH_SIZE_34
from resnet_def import model_resnet_18, model_resnet_34
from plainet_def import model_plainet_18, model_plainet_34
from common_train import training_callback, split_ds_training
from visualization import plot_histories_err, plot_val_err

current_time = datetime.now()

# Part0: split data
norm_train_ds18, norm_val_ds18 = split_ds_training(batch_size=BATCH_SIZE_18)
AUTOTUNE = tf.data.AUTOTUNE
norm_train_ds18 = norm_train_ds18.cache().prefetch(buffer_size=AUTOTUNE)
norm_val_ds18 = norm_val_ds18.cache().prefetch(buffer_size=AUTOTUNE)

norm_train_ds34, norm_val_ds34 = split_ds_training(batch_size=BATCH_SIZE_34)
AUTOTUNE = tf.data.AUTOTUNE
norm_train_ds34 = norm_train_ds34.cache().prefetch(buffer_size=AUTOTUNE)
norm_val_ds34 = norm_val_ds34.cache().prefetch(buffer_size=AUTOTUNE)

# Part1: TRAINNING OF PLAIN DEEP CNN
# plain CNN 18
plainet18 = model_plainet_18
print('Current Training plainet18 is plain net 18.')
plainet18.compile(
    optimizer='adam', # optimizer
    loss='categorical_crossentropy', # loss function to optimize
    metrics=['accuracy'] # metrics to monitor
)
start = time.time()
history18 = model_plainet_18.fit(
    norm_train_ds18,
    validation_data=norm_val_ds18,
    callbacks=training_callback,
    epochs = 20)
stop = time.time()
print(f'Training took: {(stop-start)/60} minutes')
timestamp = current_time.strftime("%m%d_%H%M")
plainet18.save(f'./maki{timestamp}_model_plainet_18.keras')


# plain CNN 34
plainet34 = model_plainet_34
print(f'Current Training Model is {model_plainet_34.name}')
plainet34.compile(
    optimizer='adam', # optimizer
    loss='categorical_crossentropy', # loss function to optimize
    metrics=['accuracy'] # metrics to monitor
)
history34 = model_plainet_34.fit(
    norm_train_ds34,
    validation_data=norm_val_ds34,
    callbacks=training_callback,
    epochs = 30) # check the model with 5 times
stop = time.time()
print(f'Training took: {(stop-start)/60} minutes')

plainet34.save(f'./maki{timestamp}_model_plainet_34.keras')
plot_histories_err(history18, history34, ['plain_cnn_18', 'plain_cnn_34'])


# Part2: TRAINNING OF ResNet CNN
resnet18 = model_resnet_18
print('Current Training model18 is resnet 18.')

resnet18.compile(
    optimizer='adam', # optimizer
    loss='categorical_crossentropy', # loss function to optimize
    metrics=['accuracy'] # metrics to monitor
)

start = time.time()
history_resnet18 = resnet18.fit(
    norm_train_ds18,
    validation_data=norm_val_ds18,
    callbacks=training_callback,
    epochs = 20)
stop = time.time()
print(f'Training took: {(stop-start)/60} minutes')

timestamp = current_time.strftime("%m%d_%H%M")
resnet18.save(f'./maki{timestamp}_model_resnet_18.keras')


# resnet 34
resnet34 = model_resnet_34
print(f'Current Training Model is {model_resnet_34.name}')
resnet34.compile(
    optimizer='adam', # optimizer
    loss='categorical_crossentropy', # loss function to optimize
    metrics=['accuracy'] # metrics to monitor
)

history_resnet34 = resnet34.fit(
    norm_train_ds34,
    validation_data=norm_val_ds34,
    callbacks=training_callback,
    epochs = 30) # check the model with 5 times

stop = time.time()
print(f'Training took: {(stop-start)/60} minutes')

resnet34.save(f'./maki{timestamp}_model_resnet_34.keras')
plot_histories_err(history_resnet18, history_resnet34, ['resnet_18', 'resnet_34'])


# Comparison
plot_val_err(history_resnet18, history_resnet34, history18, history34, ['resnet_18', 'resnet_34','plain_net_18', 'plain_net_34'], 1)
