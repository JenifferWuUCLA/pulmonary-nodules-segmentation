# -*- coding: utf-8 -*-
# generate_caffe_val_file.py - Removes the header from annotations.csv file in the val directory

import csv
import os
import shutil

############
#
# TIANCHI CSV
TIANCHI_val_path = "/home/ucla/Downloads/Caffe_CNN_Data/csv/pred/"
# TIANCHI_val_path = "/home/jenifferwu/IMAGE_MASKS_DATA/csv/val/"
TIANCHI_val_annotations = TIANCHI_val_path + "annotations.csv"

output_path = "/home/ucla/Downloads/Caffe_CNN_Data/"
# output_path = "/home/jenifferwu/Caffe_CNN_Data"
val_file = "val_0.txt"

csvRows = []

original_data_path = "/root/code/Data/"
# original_data_path = "/home/jenifferwu/IMAGE_MASKS_DATA/JPEG/Dev/"
val_data_path = "/root/code/Pulmonary_nodules_data/val/"
# val_data_path = "/home/jenifferwu/IMAGE_MASKS_DATA/JPEG/Pulmonary_nodules_data/val/"

X_avg_error_ratio = 3.191662514630957
Y_avg_error_ratio = 2.331546809978415
Z_avg_error_ratio = 0.9738836034991584
diam_avg_error_ratio = 1.0494980892679413

#####################
def csv_row(seriesuid, nodule_class):
    new_row = []
    seriesuid_list = seriesuid.split('/')
    subset, series_uid = seriesuid_list[0], seriesuid_list[1]
    re_series_uid = series_uid.replace("nodule_images_LKDS-", "_")
    val_dir, image_file, image_path = "", "", ""
    if nodule_class == 0:
        # val_dir = "n01440010/"
        image_file = "PULMONARY_NODULES_pred" + re_series_uid + ".jpg"
        image_path = val_dir + image_file
    elif nodule_class == 1:
        # val_dir = "n01440011/"
        image_file = "PULMONARY_NODULES_pred" + re_series_uid + ".jpg"
        image_path = val_dir + image_file
    new_row.append(image_path)
    # new_row.append(diameter_mm)
    new_row.append(nodule_class)
    csvRows.append(new_row)

    original_image = original_data_path + subset + "/" + series_uid + ".jpg"
    # print("original_image: %s" % str(original_image))
    tmp_image = val_data_path + val_dir + series_uid + ".jpg"
    val_image = val_data_path + val_dir + image_file

    shutil.copy(original_image, val_data_path + val_dir)
    shutil.move(tmp_image, val_image)


def is_nodule(X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio):
    # ０：不是真正肺结节；１：是真正肺结节。
    nodule_class = 0
    # print float(avg_error)
    # print float(avg_error) >= 10
    if abs(float(X_error_ratio)) <= float(X_avg_error_ratio) \
            and abs(float(Y_error_ratio)) <= float(Y_avg_error_ratio) \
            and abs(float(Z_error_ratio)) <= float(Z_avg_error_ratio) \
            and abs(float(diam_error_ratio)) <= float(diam_avg_error_ratio):
        nodule_class = 1
    return nodule_class


#####################

# Read the annotations CSV file in (skipping first row).

csvFileObj = open(TIANCHI_val_annotations)
readerObj = csv.DictReader(csvFileObj)

# csv_row('seriesuid', 'diameter_mm', 'nodule_class')
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row

    csv_row(row['seriesuid'], is_nodule(row['X_error_ratio'], row['Y_error_ratio'], row['Z_error_ratio'], row['diam_error_ratio']))

csvFileObj.close()


# Write out the val.txt CSV file.
csvFileObj = open(os.path.join(output_path, val_file), 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()
