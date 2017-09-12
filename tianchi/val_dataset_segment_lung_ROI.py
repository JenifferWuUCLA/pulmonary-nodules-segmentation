from __future__ import print_function

import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import os
import csv
import cv2
import scipy.ndimage


# out_subset = "z-nerve"
output_path = "/home/ucla/Downloads/tianchi-2D/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

tmp_workspace = os.path.join(output_path, "val/")
tmp_jpg_workspace = os.path.join(output_path, "ROI/val/")


###################################################################################

csvRows = []


def csv_row(seriesuid, imgs_mask_val):
    new_row = []
    # new_row.append(index)
    new_row.append(seriesuid)
    new_row.append(imgs_mask_val)
    csvRows.append(new_row)


###################################################################################

val_images = glob(os.path.join(output_path, "val/images_*.npy"))
for img_file in val_images:
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    imgs_to_process = np.load(img_file).astype(np.float64)
    print("on val image: %s" % img_file)
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        # Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400, 100:400]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the
        # underflow and overflow on the pixel spectrum
        img[img == max] = mean
        img[img == min] = mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
        #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region
        # engulf the vessels and incursions into the lung cavity by
        # radio opaque tissue
        #
        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))
        #
        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions
        #  with a bbox that is to large in either dimnsion
        #  Also, the lungs are generally far away from the top
        #  and bottom of the image, so any regions that are too
        #  close to the top and bottom are removed
        #  This does not produce a perfect segmentation of the lungs
        #  from the image, but it is surprisingly good considering its
        #  simplicity.
        #
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        mask = np.ndarray([512, 512], dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodules
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask
        #
        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
        imgs_to_process[i] = mask
    np.save(img_file.replace("images", "lungmask"), imgs_to_process)

#
#    Here we're applying the masks and cropping and resizing the image
#


val_images = glob(os.path.join(output_path, "val/lungmask_*.npy"))
out_images = []  # final set of images
out_nodule_masks = []  # final set of nodule_masks
seriesuids = []
for fname in val_images:
    print("working on val lung mask file: %s" % fname)
    imgs_to_process = np.load(fname.replace("lungmask", "images"))
    masks = np.load(fname)
    nodule_masks = np.load(fname.replace("lungmask", "masks"))
    for i in range(len(imgs_to_process)):
        mask = masks[i]
        nodule_mask = nodule_masks[i]
        img = imgs_to_process[i, :, :]
        # new_size = [512, 512]  # we're scaling back up to the original size of the image

        slice = img * mask        # apply lung mask
        filename = fname.replace(tmp_workspace, "").replace("lungmask", "lung_images")
        new_lung_name = filename.replace(".npy", "") + "_%s.jpg" % (i)
        image_path = tmp_jpg_workspace
        print(new_lung_name, image_path)
        cv2.imwrite(os.path.join(image_path, new_lung_name), slice)

        nodule_slice = img * nodule_mask
        filename = fname.replace(tmp_workspace, "").replace("lungmask", "nodule_images")
        new_nodule_name = filename.replace(".npy", "") + "_%s.jpg" % (i)
        image_path = tmp_jpg_workspace
        print(new_nodule_name, image_path)
        cv2.imwrite(os.path.join(image_path, new_nodule_name), nodule_slice)

        #
        # renormalizing the masked image (in the mask region)
        #
        new_mean = np.mean(slice[mask > 0])
        new_std = np.std(slice[mask > 0])
        #
        #  Pulling the background color up to the lower end
        #  of the pixel range for the lungs
        #
        old_min = np.min(slice)  # background color
        slice[slice == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
        slice = slice - new_mean
        slice = slice / new_std
        # make image bounding box  (min row, min col, max row, max col)
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        #
        # Finding the global min and max row over all regions
        #
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col - min_col
        height = max_row - min_row
        if width > height:
            max_row = min_row + width
        else:
            max_col = min_col + height
        #
        # cropping the image down to the bounding box for all regions
        # (there's probably an skimage command that can do this in one line)
        #
        slice = slice[min_row:max_row, min_col:max_col]
        mask = mask[min_row:max_row, min_col:max_col]
        if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
            pass
        else:
            # moving range to -1 to 1 to accomodate the resize function
            mean = np.mean(slice)
            slice = slice - mean
            min = np.min(slice)
            max = np.max(slice)
            slice = slice / (max - min)

            nodule_mask = scipy.ndimage.interpolation.zoom(nodule_mask, [1.0, 1.0], mode='nearest')
            nodule_mask[nodule_mask < 0.5] = 0
            nodule_mask[nodule_mask > 0.5] = 1
            nodule_mask = nodule_mask.astype('int8')
            nodule_mask = 255.0 * nodule_mask
            nodule_mask = nodule_mask.astype(np.uint8)

            new_img = resize(slice, [512, 512])
            new_nodule_mask = resize(nodule_mask[min_row:max_row, min_col:max_col], [512, 512])

            filename = fname.replace(tmp_workspace, "").replace("lungmask", "nodule_pred_mask")
            nodule_pred_name = filename.replace(".npy", "") + "_%s.jpg" % (i)
            image_path = tmp_jpg_workspace
            print(nodule_pred_name, image_path)
            cv2.imwrite(os.path.join(image_path, nodule_pred_name), new_nodule_mask)

            out_images.append(new_img)
            out_nodule_masks.append(new_nodule_mask)

            image_path = fname.replace("lungmask", "images")
            image_name = image_path.replace(os.path.join(output_path, "val/"), "").replace("images_", "") + "_%s" % (i)
            seriesuids.append(image_name.replace(".npy", ""))

num_images = len(out_images)
#
#  Writing out images and masks as 1 channel arrays for input into network
#
final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
for i in range(num_images):
    final_images[i, 0] = out_images[i]
    final_masks[i, 0] = out_nodule_masks[i]

rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
# test_i = int(0.2*num_images)

np.save(os.path.join(output_path, "val/valImages.npy"), final_images[rand_i[:]])
np.save(os.path.join(output_path, "val/valMasks.npy"), final_masks[rand_i[:]])

csv_row("seriesuid", "pred_image")
for i in range(num_images):
    index = rand_i[i]
    seriesuid = seriesuids[index]
    imgs_mask_test = 'imgs_mask_test_%04d.npy' % (i)
    csv_row(seriesuid, imgs_mask_test)

# Write out the imgs_mask_val_coordinate CSV file.
pred_image_file = "seriesuid_pred_image.csv"
print(os.path.join(output_path, pred_image_file))
csvFileObj = open(os.path.join(output_path, pred_image_file), 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()