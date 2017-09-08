from __future__ import print_function

import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from glob import glob
import os
import cv2
import scipy.ndimage


# out_subset = "z-nerve"
output_path = "/home/ucla/Downloads/tianchi-2D/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

tmp_workspace = os.path.join(output_path, "train/")
tmp_jpg_workspace = os.path.join(output_path, "ROI/train/")


train_images = glob(os.path.join(output_path, "train/images_*.npy"))
for img_file in train_images:
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    imgs_to_process = np.load(img_file).astype(np.float64)
    print("on train image: %s" % img_file)
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        #Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400,100:400] 
        mean = np.mean(middle)  
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the 
        # underflow and overflow on the pixel spectrum
        img[img==max]=mean
        img[img==min]=mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid 
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold, 1.0, 0.0)  # threshold the image
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


train_images = glob(os.path.join(output_path, "train/lungmask_*.npy"))
out_images = []      #final set of images
out_nodule_masks = []   #final set of nodule_masks
for fname in train_images:
    print("working on train lung mask file: %s" % fname)
    imgs_to_process = np.load(fname.replace("lungmask", "images"))
    masks = np.load(fname)
    nodule_masks = np.load(fname.replace("lungmask", "masks"))
    for i in range(len(imgs_to_process)):
        mask = masks[i]
        nodule_mask = nodule_masks[i]
        img = imgs_to_process[i, :, :]
        # new_size = [512, 512]   # we're scaling back up to the original size of the image

        slice = img * mask         # apply lung mask
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

        nodule_mask = scipy.ndimage.interpolation.zoom(nodule_mask, [1.0, 1.0], mode='nearest')
        nodule_mask[nodule_mask < 0.5] = 0
        nodule_mask[nodule_mask > 0.5] = 1
        nodule_mask = nodule_mask.astype('int8')
        nodule_mask = 255.0 * nodule_mask
        nodule_mask = nodule_mask.astype(np.uint8)

        new_img = slice
        new_nodule_mask = nodule_mask

        filename = fname.replace(tmp_workspace, "").replace("lungmask", "nodule_pred_mask")
        nodule_pred_name = filename.replace(".npy", "") + "_%s.jpg" % (i)
        image_path = tmp_jpg_workspace
        print(nodule_pred_name, image_path)
        cv2.imwrite(os.path.join(image_path, nodule_pred_name), new_nodule_mask)

        out_images.append(new_img)
        out_nodule_masks.append(new_nodule_mask)

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

np.save(os.path.join(output_path, "train/trainImages.npy"), final_images[rand_i[:]])
np.save(os.path.join(output_path, "train/trainMasks.npy"), final_masks[rand_i[:]])