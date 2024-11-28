from globalvar import *

## step 2-1: define building blocks of data standardization and augmentation
# to reduce overfitting

# from paper[1]; 3.4 Implementation paragraph 1;
# the image is resized with its shorter side randomly sampled in [256,480] for scale augmentation
# as this should be before augmentation methods
def _resize_image(im: PIL.Image) -> PIL.Image:
    width, height = im.size
    target_size = random.randint(256, 480)
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return im.resize((new_width, new_height), PIL.Image.BICUBIC)

# from paper[2] 4.1;
# function that takes an PIL.Image in and laters the intensities of the RGB channels, 
# specifically perform PCA on the set of RGB pixel values throughout the ImageNet training set 
# to each training image, add multiples of the found principal components, 
# with magnitudes proportional to the corresponding eigenvalues times a random variable drawn
# from a Gaussian with mean zero and standard deviation 0.1
def _augmentation_rgb(image: PIL.Image) -> PIL.Image:

    img_array = np.array(image)
    # print(f'img size of {img_array.shape}')
    pixels = img_array.reshape(-1, 3)
    # perform PCA on RGB pixels
    mean = np.mean(pixels, axis=0)
    centered_pixels = pixels - mean
    cov_matrix = np.cov(centered_pixels.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # adjust original images
    random_values = np.random.normal(0, 0.1, 3)
    adjustments = np.sum(eigenvectors * eigenvalues * random_values[:, np.newaxis], axis=1)
    altered_pixels = pixels + adjustments
    # ensure in the valid range [0, 255]
    altered_pixels = np.clip(altered_pixels, 0, 255).astype(np.uint8)
    # shape back
    altered_img_array = altered_pixels.reshape(patch_size, patch_size, 3)
    # altered_img_array = altered_pixels.reshape(img_array.shape)
    altered_image = PIL.Image.fromarray(altered_img_array)
    return altered_image

# from paper[2] 4.1
# the first form of data augmentation that is flig images and have 224*224 patches 
def _augmentation_crop(im: PIL.Image) -> list[PIL.Image]:
    if im.mode != 'RGB':
        im = im.convert('RGB')
    result = []
    flipped = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    width, height = im.size
    if width < patch_size or height < patch_size:
        printf("warining: image too small for 224*224 crop, skip it")
        return []

    # im and flipped * 1
    for i in range(2):
        source = [im, flipped][i]
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        patch = source.crop((left, top, left + patch_size, top + patch_size))
        result.append(patch)
    return result

# step 2-2: define and run the whole process of data standardization and augmentation
# to reduce overfitting
# after processing the input data, we will get
# - middle_folder created with cropped augmented images
# - out_folder created with images from middle_folder after normalization

# TODO: maki, the question here is we augment both training & validating data, and use raw test img ?
# TODO: maki, or the
# relies on the global variables: in_folder,im_per_class
# returns the mean image
import random

# generate middle_img
def preprocess_data() -> PIL.Image:
    sum_image = None
    total_images = 0
    for fld in os.listdir(in_folder):
        out = os.path.join(middle_folder, fld)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        fld_path = pathlib.Path(os.path.join(in_folder, fld))
        num_images = 0
        img_id = 0
        for file in list(fld_path.glob('*')):
            with PIL.Image.open(file) as im:
                resized_img = _resize_image(im)
                cropped_img_list = _augmentation_crop(resized_img)
                for cropped_img in cropped_img_list:
                    img_id += 1
                    rgb_img = _augmentation_rgb(cropped_img)
                    rgb_img.save(os.path.join(out, str(img_id) + '.jpg'))
                    img_array = np.array(rgb_img, dtype=np.float64)
                    if sum_image is None:
                        sum_image = img_array
                    else:
                        sum_image += img_array
                    total_images += 1
                    if total_images % 1000 == 0:
                        print(f'preprocessed {total_images} data')
            num_images += 1
            # we use all the data instead of strict balanced data
            # if num_images >= im_per_class:
            #     break
    # Calculate mean image
    if total_images > 0:
        mean_image = (sum_image / total_images).astype(np.uint8)
    else:
        raise ValueError("No images processed")
    return PIL.Image.fromarray(mean_image)

# both paper[1] and paper[2] mentions normalization to subtract mean
# todo: maki comment, I think mean should be calculated after the images are of same size
# todo: how about we put it in the last step
# we use same data size of each calss
# relies on the global variables: middle_folder, out_folder, im_per_class
def normalize_data():
    print(f'start data normalization')
    for fld in os.listdir(middle_folder):
        out = os.path.join(out_folder, fld)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        fld_path = pathlib.Path(os.path.join(middle_folder, fld))
        num_images = 0
        for file in list(fld_path.glob('*')):
            with PIL.Image.open(file) as im:
                im_arr = np.array(im)
                result_arr = im_arr - mean_img_arr
                result_arr = np.clip(result_arr, 0, 255)
                result_img = PIL.Image.fromarray(result_arr)
                result_img.save(os.path.join(out, str(num_images)+'.jpg'))
            num_images += 1

start = time.time()
mean_img = preprocess_data()
mean_img_path = os.path.join(main_folder, 'mean_img.jpg')
mean_img.save(mean_img_path)
mean_img_arr = np.array(mean_img)
normalize_data()
stop = time.time()
print(f'Data Augementation and Normalization took: {(stop-start)/60} minutes')
# after this step, the
