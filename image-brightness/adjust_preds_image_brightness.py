import cv2
from glob import glob
import os

# out_subset = "z-nerve/"
output_path = "/home/ucla/Downloads/tianchi-2D/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

test_images = glob(output_path + "image-coordinate-2D/*.jpg")


if __name__ == '__main__':
    index = 0
    for fn in test_images:
        print('loading %s ...' % fn)
        print('processing...')
        image_name = fn.replace(output_path + "image-coordinate-2D/", "")
        print("image_name: %s" % image_name)
        img = cv2.imread(fn)
        w = img.shape[1]
        h = img.shape[0]
        ii = 0
        # let image get darker
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                # set the pixel value decrease to 20%
                img[xj, xi, 0] = int(img[xj, xi, 0] * 100)
                img[xj, xi, 1] = int(img[xj, xi, 1] * 100)
                img[xj, xi, 2] = int(img[xj, xi, 2] * 100)
        # cv2.imshow('img', img)
        # cv2.imwrite(os.path.join(output_path + "image-coordinate-2D/bright_01/", 'imgs_mask_test_%04d.jpg' % (index)), img)
        cv2.imwrite(os.path.join(output_path + "image-coordinate-2D/bright_01/", '%s' % (image_name)), img)
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                ##set the pixel value increase to 1020%
                img[xj, xi, 0] = int(img[xj, xi, 0] * 1000.2)
                img[xj, xi, 1] = int(img[xj, xi, 1] * 1000.2)
                img[xj, xi, 2] = int(img[xj, xi, 2] * 1000.2)
        # cv2.imshow('img',img)
        # cv2.imwrite(os.path.join(output_path + "image-coordinate-2D/bright_02/", 'imgs_mask_test_%04d.jpg' % (index)), img)
        cv2.imwrite(os.path.join(output_path + "image-coordinate-2D/bright_02/", '%s' % (image_name)), img)
        index += 1
