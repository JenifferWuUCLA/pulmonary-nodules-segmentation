from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from glob import glob

# data_path = 'raw/'

image_rows = 420
image_cols = 580

# subset = "train_dataset/"
subset = "data_set/"
# tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
tianchi_path = "/home/jenifferwu/LUNA2016/"
tianchi_subset_path = tianchi_path + subset

out_subset = "nerve"
output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset


def create_train_data():
    train_data_path = os.path.join(tianchi_subset_path, 'train')
    # images = os.listdir(train_data_path)
    images = glob(train_data_path + "*.mhd")
    total = len(images) / 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(output_path, "train/imgs_train.npy"), imgs)
    np.save(os.path.join(output_path, "train/imgs_mask_train.npy"), imgs_mask)
    # np.save('imgs_train.npy', imgs)
    # np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load(os.path.join(output_path, "train/imgs_train.npy"))
    imgs_mask_train = np.load(os.path.join(output_path, "train/imgs_mask_train.npy"))
    return imgs_train, imgs_mask_train


def create_test_data():
    test_data_path = os.path.join(tianchi_subset_path, 'test')
    print("test_data_path: %s" % test_data_path)
    # train_data_path = os.path.join(data_path, 'test')
    # images = os.listdir(test_data_path)
    images = glob(test_data_path + "*.mhd")
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    for image_name in images:
        print("image_name: %s" % image_name)
        img_id = int(image_name.split('.')[0])
        print("img_id: %s" % img_id)
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(output_path, "test/imgs_test.npy"), imgs)
    np.save(os.path.join(output_path, "test/imgs_id_test.npy"), imgs_id)
    # np.save('imgs_test.npy', imgs)
    # np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load(os.path.join(output_path, "test/imgs_test.npy"))
    imgs_id = np.load(os.path.join(output_path, "test/imgs_id_test.npy"))
    return imgs_test, imgs_id


if __name__ == '__main__':
    create_train_data()
    create_test_data()
