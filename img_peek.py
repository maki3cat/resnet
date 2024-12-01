"""print samples of dataset"""

import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Define the main directory and subdirectories
main_dir = 'data/cooked-img'
subdirs = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
           'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# Function to get random images from subdirectories
def get_random_images(num_images):
    images = []
    for _ in range(num_images):
        subdir = random.choice(subdirs)
        subdir_path = os.path.join(main_dir, subdir)
        if os.path.exists(subdir_path):
            files = os.listdir(subdir_path)
            if files:
                img_file = random.choice(files)
                img_path = os.path.join(subdir_path, img_file)
                img = Image.open(img_path)
                images.append(img)
    return images

# Get 16 random images
images = get_random_images(16)

# Create a 4x4 grid to display the images
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

# Plot each image in the grid
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
    ax.axis('off')

plt.tight_layout()
plt.show()
