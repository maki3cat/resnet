from globalvar import *
from plainet_def import *
from common_train import *
from visualization import *

current_time = datetime.now()

# MODEL 18
model18 = model_plainet_18
print(f'Current Training model18 is plain net 18.')

model18.compile(
    optimizer='adam', # optimizer
    loss='categorical_crossentropy', # loss function to optimize 
    metrics=['accuracy'] # metrics to monitor
)

import time
start = time.time()
history18 = model18.fit(
    norm_train_ds18,
    validation_data=norm_val_ds18,
    callbacks=callbacks,
    epochs = 20)
stop = time.time()
print(f'Training took: {(stop-start)/60} minutes')

timestamp = current_time.strftime("%m%d_%H%M")
model18.save(f'./maki{timestamp}_model_plainet_18.keras')


# MODEL 34
model34 = model_plainet_34
print(f'Current Training Model is {plainet_model_34.name}')
model34.compile(
    optimizer='adam', # optimizer
    loss='categorical_crossentropy', # loss function to optimize 
    metrics=['accuracy'] # metrics to monitor
)

history34 = model34.fit(
    norm_train_ds34,
    validation_data=norm_val_ds34,
    callbacks=callbacks,
    epochs = 30) # check the model with 5 times

stop = time.time()
print(f'Training took: {(stop-start)/60} minutes')

model34.save(f'./maki{timestamp}_model_plainet_34.keras')

plot_histories(history18, history34, ['plain_net_18', 'plain_net_34'])
