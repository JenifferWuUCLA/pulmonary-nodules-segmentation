from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
import os
import cv2


# subset = "server-test/"
output_path = "/home/ucla/Downloads/tianchi-Segmentation/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + subset


K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


# ########################################U-Net Segmentation Prediction IoU Start######################################
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def computeIoU(y_true_batch, y_pred_batch):
    return np.mean(np.asarray([pixelAccuracy(y_true_batch[i], y_pred_batch[i]) for i in range(len(y_true_batch))]))


def pixelAccuracy(y_true, y_pred):
    y_true = np.argmax(np.reshape(y_true, [img_rows, img_cols]), axis=0)
    y_pred = np.argmax(np.reshape(y_pred, [img_rows, img_cols]), axis=0)
    y_pred = y_pred * (y_true > 0)

    return 1.0 * np.sum((y_pred == y_true) * (y_true > 0)) / np.sum(y_true > 0)


def intersect(im1, im2):
    """ return the intersection of two lists """
    print("im1[0][0][0]: ")
    print(im1[0][0][0])
    print("im2[0][0][0]: ")
    print(im2[0][0][0])
    return list(set(im1[0][0][0]) & set(im2[0][0][0]))


def union(im1, im2):
    """ return the union of two lists """
    return list(set(im1[0][0][0]) | set(im2[0][0][0]))


def intersectOverUnion(I, U):
    """ return the intersection over union of two lists """
    return len(I)/float(len(U))

# ########################################U-Net Segmentation Prediction IoU End######################################


def get_net(load_weight_path=None):
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

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

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

    pred_dir = os.path.join(output_path, 'data_images/pred-images/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for i in range(num_test):
        np.save(os.path.join(output_path + "data_images/preds/", 'imgs_mask_test_%04d.npy' % (i)), imgs_mask_test[i, 0])
        cv2.imwrite(os.path.join(pred_dir, 'imgs_mask_test_%04d.jpg' % (i)), imgs_mask_test[i, 0])

    print("====================================U-Net Segmentation Prediction IoU======================================")
    mean = 0.0
    for i in range(num_test):
        mean += dice(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
    mean /= num_test
    print("Mean Dice Coeff : ", mean)

    # mean = 0.0
    # for i in range(num_test):
    # mean += computeIoU(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
    # print("imgs_mask_test_true: ")
    # print(imgs_mask_test_true)
    # print("imgs_mask_test: ")
    # print(imgs_mask_test)
    I = intersect(imgs_mask_test_true, imgs_mask_test)
    U = union(imgs_mask_test_true, imgs_mask_test)
    IoU = intersectOverUnion(I, U)
    print("Intersection Over Union : ", IoU)


if __name__ == '__main__':
    predict()
