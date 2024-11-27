from globalvar import *
from checkenv import *
from resnetmdl import model


# PART-A: DEFINITION OF THE MODEL
data_dir = pathlib.Path(out_folder)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="training",
    label_mode='categorical', # default mode is 'int' label, but we want one-hot encoded labels (e.g. for categorical_crossentropy loss)
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    label_mode='categorical',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# i = 1
# for images, labels in train_ds.take(1):
#     for (image, label) in zip(images, labels): 
#         ax = plt.subplot(4, 4, i)
#         plt.imshow(image.numpy().astype("uint8"))
#         plt.title(class_names[tf.argmax(label, axis=0)])
#         plt.axis("off")
#         i = i + 1
#         if i == 17:
#             break
# plt.show()

# use keras functionality for adding a rescaling layer
# maki: this api is deprecated
# normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalization_layer = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

# rescale training and validation sets
norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(norm_train_ds))
# get one image
first_image = image_batch[0]
# confirm pixel values are now in the [0,1] range
print(np.min(first_image), np.max(first_image))


# PART-B: TRAINING OF THE MODEL
model.compile(
    optimizer='adam', # optimizer
    loss='categorical_crossentropy', # loss function to optimize 
    metrics=['accuracy'] # metrics to monitor
)

AUTOTUNE = tf.data.AUTOTUNE
norm_train_ds = norm_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
norm_val_ds = norm_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", # monitor validation loss (that is, the loss computed for the validation holdout)
        min_delta=1e-2, # "no longer improving" being defined as "an improvement lower than 1e-2"
        patience=10, # "no longer improving" being further defined as "for at least 10 consecutive epochs"
        verbose=1
    )
]
import time
start = time.time()
model.fit(
    norm_train_ds,
    validation_data=norm_val_ds,
    callbacks=callbacks,
    epochs = 20)

stop = time.time()

print(f'Training took: {(stop-start)/60} minutes')
