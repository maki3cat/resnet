from globalvar import *
from plainet_def import *

class_names = []

def split_ds_training(batch_size):
    data_dir = pathlib.Path(train_folder)
    # training and validation
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        label_mode='categorical',
        seed=100,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        label_mode='categorical',
        seed=100,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    class_names.extend(train_ds.class_names)
    # Create normalization layer
    normalization_layer = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255)
    ])

    # Normalize datasets
    norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Verify normalization (optional)
    image_batch, labels_batch = next(iter(norm_train_ds))
    first_image = image_batch[0]
    print("Pixel value range:", np.min(first_image), "-", np.max(first_image))
    return norm_train_ds, norm_val_ds

norm_train_ds18, norm_val_ds18 = split_ds_training(batch_size=batch_size_18)
AUTOTUNE = tf.data.AUTOTUNE
norm_train_ds18 = norm_train_ds18.cache().prefetch(buffer_size=AUTOTUNE)
norm_val_ds18 = norm_val_ds18.cache().prefetch(buffer_size=AUTOTUNE)

norm_train_ds34, norm_val_ds34 = split_ds_training(batch_size=batch_size_34)
AUTOTUNE = tf.data.AUTOTUNE
norm_train_ds34 = norm_train_ds34.cache().prefetch(buffer_size=AUTOTUNE)
norm_val_ds34 = norm_val_ds34.cache().prefetch(buffer_size=AUTOTUNE)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=10,
        verbose=1
    )
]
print("Class names:", class_names)

