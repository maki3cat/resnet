from checkenv import *
from globalvar import in_folder, out_folder

# TODO: maki, check if this matches our preprocessing of image in report
# TODO: should use a uniform size like the paper
def square_crop_image(im: PIL.Image) -> PIL.Image:
    width, height = im.size
    new_size = min(width, height)
    # center crop
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    crop_im = im.crop((left, top, right, bottom))
    crop_im = crop_im.convert('RGB')
    return crop_im

def make_dataset(in_folder, im_per_class):
    # iterate through all folders (there should be one folder per object class)
    for fld in os.listdir(in_folder):
        # create the output folder for processed images for current class
        # delete folder and contents if there is one already
        out = os.path.join(out_folder, fld)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)

        fld_path = pathlib.Path(os.path.join(in_folder, fld))
        num_images = 0
        for file in list(fld_path.glob('*')):
            # open image, center crop to a square
            # save to the output folder
            with PIL.Image.open(file) as im:
                crop_im = square_crop_image(im)
                crop_im.save(os.path.join(out, str(num_images) + '.jpg'))
                # im.close(), with automatically closes

            # break when desired number of images
            # has been processed (to keep classes balance)
            num_images = num_images + 1
            if (num_images > im_per_class):
                break

file_count = []
# get number of images in each folder (images per class)
for fld in os.listdir(in_folder):
    crt = os.path.join(in_folder, fld)
    image_count = len(os.listdir(crt))
    file_count.append(image_count)
    print(f'{crt} contains {image_count} images')

# get the number of images that will make our classes balanced
im_per_class = min(file_count)
# process input images
make_dataset(in_folder, im_per_class)

# img_height = image_size[1]
# img_width = image_size[0]
# batch_size = 32
#
# data_dir = pathlib.Path(out_folder)
#
