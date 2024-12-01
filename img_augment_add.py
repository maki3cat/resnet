"""Module for additional image augmentation techniques that have not been integrated."""

import numpy as np
import PIL.Image
from config import PATCH_SIZE

# from paper[2] 4.1;
# function that takes an PIL.Image in and laters the intensities of the RGB channels,
# specifically perform PCA on the set of RGB pixel values throughout the ImageNet training set
# to each training image, add multiples of the found principal components,
# with magnitudes proportional to the corresponding eigenvalues times a random variable drawn
# from a Gaussian with mean zero and standard deviation 0.1
# TODO: this is not applied
def _augmentation_rgb(im: PIL.Image) -> PIL.Image:
    img_array = np.array(im)
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
    altered_img_array = altered_pixels.reshape(PATCH_SIZE, PATCH_SIZE, 3)
    # altered_img_array = altered_pixels.reshape(img_array.shape)
    altered_image = PIL.Image.fromarray(altered_img_array)
    return altered_image
