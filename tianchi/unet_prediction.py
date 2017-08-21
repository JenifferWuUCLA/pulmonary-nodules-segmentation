from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
import os
import cv2


subset = "server-test-2D/"
output_path = "/home/ucla/Downloads/tianchi-2D/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + subset


K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

CUBE_SIZE = 32


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False, mal=False):
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def predict():
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test = np.load(os.path.join(output_path, "val/valImages.npy")).astype(np.float32)
    imgs_mask_test_true = np.load(os.path.join(output_path, "val/valMasks.npy")).astype(np.float32)

    # loading best weights from training session
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    load_weight_path = os.path.join(output_path, 'preds/') + 'unet.hdf5'
    model = get_net(load_weight_path)

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    num_test = len(imgs_test)
    print("num_test: %d" % num_test)
    imgs_mask_test = np.ndarray([num_test, 1, 512, 512], dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i + 1]], verbose=0)[0]
    # np.save(os.path.join(output_path + "data_images/preds/", "masksTestPredicted.npy"), imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    pred_dir = os.path.join(output_path, 'data_images/pred-images/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for i in range(num_test):
        np.save(os.path.join(output_path + "data_images/preds/", 'imgs_mask_test_%04d.npy' % (i)), imgs_mask_test[i, 0])
        cv2.imwrite(os.path.join(pred_dir, 'imgs_mask_test_%04d.jpg' % (i)), imgs_mask_test[i, 0])

    mean = 0.0
    for i in range(num_test):
        mean += dice_coef_np(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
    mean /= num_test
    print("Mean Dice Coeff : ", mean)


if __name__ == '__main__':
    predict()
