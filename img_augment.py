
import os
import random
import pathlib
import numpy as np
import PIL.Image
import tensorflow as tf
from config import in_folder, out_folder, resized_data_folder, PATCH_SIZE, AUGMENT_MULTIPLIER

# from paper[1]; 3.4 Implementation paragraph 1;
# the image is resized with its shorter side randomly sampled in [256,480] for scale augmentation
# as this should be before augmentation methods
def _resize_to_rgb(im: PIL.Image) -> PIL.Image:
    im_array = np.array(im)
    im_tensor = tf.convert_to_tensor(im_array)
    if len(im_tensor.shape) < 3 or im_tensor.shape[-1] not in [1, 3]:
        return None
    to_size = (random.randint(256, 480), random.randint(256, 480))
    re_tensor = tf.image.resize(im_tensor, to_size, method=tf.image.ResizeMethod.BICUBIC)
    re_im_array = tf.cast(re_tensor, tf.uint8).numpy()
    re_im = PIL.Image.fromarray(re_im_array)
    return re_im.convert('RGB')

# start from raw-img
def resize_all_raw_data():
    total = 0
    for fld in os.listdir(in_folder):
        out = os.path.join(resized_data_folder, fld)
        if os.path.exists(out):
            raise ValueError("folder exists, please check if you need recalculation")
        os.makedirs(out)
        fld_path = pathlib.Path(os.path.join(in_folder, fld))
        img_id = 0
        for file in list(fld_path.glob('*')):
            img_id += 1
            if (total % 1000) == 0:
                print(f"has resized {total} images")
            with PIL.Image.open(file) as im:
                new_img = _resize_to_rgb(im)
                if new_img:
                    new_img.save(os.path.join(out, str(img_id) + '.jpg'))
                    total += 1


# from paper[2] 4.1
# the first form of data augmentation that is flig images and have 224*224 patches, this increases interdependence, but enlarge dataset to reduce overfitting is more importantly
def _crop_gen(im: PIL.Image) -> list[PIL.Image]:
    result = []
    flipped = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    width, height = im.size
    if width < PATCH_SIZE or height < PATCH_SIZE:
        print("warining: image too small for 224*224 crop, skip it")
        return []
    # im and flipped * 1
    for i in range(AUGMENT_MULTIPLIER):
        source = [im, flipped][i%2]
        left = random.randint(0, width - PATCH_SIZE)
        top = random.randint(0, height - PATCH_SIZE)
        patch = source.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
        result.append(patch)
    return result

# start from resized-img
def crop_gen_all_data():
    total = 0
    _in = resized_data_folder
    for fld in os.listdir(_in):
        out = os.path.join(out_folder, fld)
        if not os.path.exists(out):
            os.makedirs(out)
        fld_path = pathlib.Path(os.path.join(_in, fld))
        img_id = 0
        for file in list(fld_path.glob('*')):
            if (total % 2000) == 0:
                print(f"has crop and generated {total} images")
            with PIL.Image.open(file) as im:
                new_img_list = _crop_gen(im)
                for new_img in new_img_list:
                    new_img.save(os.path.join(out, str(img_id) + '.jpg'))
                    total += 1
                    img_id += 1
