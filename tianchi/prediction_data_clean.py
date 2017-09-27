from __future__ import print_function

import numpy as np
from glob import glob
import os
import csv


# out_subset = "z-nerve/"
output_path = "/home/ucla/Downloads/tianchi-Unet/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

tmp_workspace = os.path.join(output_path, "val/")


###################################################################################
def image_file_name(o_image_name):
    image_name = o_image_name.split("_")[1]
    # v_z = o_image_name.split("_")[2]
    # Read the annotations CSV file in (skipping first row).
    if os.path.exists(os.path.join(output_path, "seriesuid_pred_image.csv")):
        csvFileObj = open(os.path.join(output_path, "seriesuid_pred_image.csv"), 'r')
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if readerObj.line_num == 1:
                continue  # skip first row
            seriesuid = row['seriesuid']
            pred_image = row['pred_image']
            if image_name == seriesuid:
                csvFileObj.close()
                return pred_image
        csvFileObj.close()

csvRows = []


def csv_row(seriesuid, imgs_mask_val):
    new_row = []
    # new_row.append(index)
    new_row.append(seriesuid)
    new_row.append(imgs_mask_val)
    csvRows.append(new_row)


###################################################################################
csv_row("seriesuid", "pred_image")

val_images = glob(os.path.join(output_path, "val/images_*.npy"))
for img_file in val_images:
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    imgs_to_process = np.load(img_file).astype(np.float64)
    print("on val image: %s" % img_file)
    o_image_name = img_file.replace(tmp_workspace, "").replace(".npy", "")
    print("o_image_name: %s" % o_image_name)

    seriesuid = o_image_name.replace("images_", "")
    pred_image = image_file_name(o_image_name)
    csv_row(seriesuid, pred_image)

# Write out the imgs_mask_val_coordinate CSV file.
pred_image_file = "seriesuid_pred_image_clean.csv"
print(os.path.join(output_path, pred_image_file))
csvFileObj = open(os.path.join(output_path, pred_image_file), 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()
