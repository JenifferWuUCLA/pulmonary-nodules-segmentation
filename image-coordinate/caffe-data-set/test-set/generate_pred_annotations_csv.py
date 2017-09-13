# -*- coding: utf-8 -*-
# generate_train_annotations_csv.py - just keep 3 slices

from __future__ import print_function, division
import os
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd

try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

############
#
# Getting list of image files
subset = "val_subset_all/"
# subset = "data_set/val/"
tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
# tianchi_path = "/home/jenifferwu/LUNA2016/"
tianchi_subset_path = tianchi_path + subset

output_path = "/home/ucla/Downloads/Caffe_CNN_Data/csv/pred/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/csv/pred/"

file_list = glob(tianchi_subset_path + "*.mhd")

csvRows = []


########################################################################################################################
#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


def csv_row(seriesuid, coordX, coordY, coordZ, diameter_mm, X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio):
    new_row = []
    new_row.append(seriesuid)
    new_row.append(coordX)
    new_row.append(coordY)
    new_row.append(coordZ)
    new_row.append(diameter_mm)
    new_row.append(X_error_ratio)
    new_row.append(Y_error_ratio)
    new_row.append(Z_error_ratio)
    new_row.append(diam_error_ratio)
    csvRows.append(new_row)


########################################################################################################################

csv_path = "/home/ucla/Downloads/tianchi-2D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
statistics_original_file = os.path.join(csv_path, "statistics_error_ratios.csv")

# Read the statistics_original.csv in (skipping first row).
stat_csvRows = []
csvFileObj = open(statistics_original_file)
readerObj = csv.reader(csvFileObj)
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row
    stat_csvRows.append(row)
csvFileObj.close()


def get_prediction_error(images_name, coordX, coordY, coordZ, diameter_mm):
    # print("images_name, coordX, coordY, coordZ, diameter_mm: ")
    # print(images_name, coordX, coordY, coordZ, diameter_mm)
    for stat_row in stat_csvRows:
        seriesuid = stat_row[0]

        pred_coordX = stat_row[1]
        pred_coordY = stat_row[2]
        pred_coordZ = stat_row[3]
        pred_diameter_mm = stat_row[4]

        X_error_ratio = stat_row[5]
        Y_error_ratio = stat_row[6]
        Z_error_ratio = stat_row[7]
        diam_error_ratio = stat_row[8]

        if seriesuid == images_name and pred_coordX == coordX and pred_coordY == coordY and pred_coordZ == coordZ and pred_diameter_mm == diameter_mm:
            # print("X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio: ")
            # print(X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio)
            return X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio


#
# The locations of the nodes
# print(tianchi_csv_path)
df_node = pd.read_csv(tianchi_path + "csv/test/annotations.csv")
# df_node = pd.read_csv(tianchi_path + "annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
# print(df_node["file"])
df_node = df_node.dropna()

#####
csv_row("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "X_error_ratio", "Y_error_ratio", "Z_error_ratio", "diam_error_ratio")

# Read the annotations CSV file in (skipping first row).
if os.path.exists(os.path.join(output_path, "annotations.csv")):
    csvFileObj = open(os.path.join(output_path, "annotations.csv"), 'r')
    readerObj = csv.DictReader(csvFileObj)
    for row in readerObj:
        if readerObj.line_num == 1:
            continue  # skip first row
        csv_row(row['seriesuid'], row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm'], row['X_error_ratio'], row['Y_error_ratio'], row['Z_error_ratio'], row['diam_error_ratio'])
    csvFileObj.close()

#
# Looping over the image files
#
for fcount, img_file in enumerate(tqdm(file_list)):
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    seriesuid = img_file.replace(tianchi_subset_path, "").replace(".mhd", "")
    # print("img_file: %s" % str(img_file))
    # print("seriesuid: %s" % str(seriesuid))
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)
        # print(mini_df)
        for node_idx, cur_row in mini_df.iterrows():
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            # just keep 3 slices
            center = np.array([node_x, node_y, node_z])  # nodule center
            v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
            # print("images_%04d_%04d.npy" % (fcount, node_idx))
            # print("masks_%04d_%04d.npy" % (fcount, node_idx))
            images_name = "nodule_images_%s" % (cur_row["seriesuid"])
            # print("images_name: %s" % images_name)
            X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio = get_prediction_error(cur_row["seriesuid"], node_x, node_y, node_z, diam)
            # print("X_error_ratio: %s, Y_error_ratio: %s, Z_error_ratio: %s, diam_error_ratio: %s: " % (X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio))
            # for i in range(3):
            csv_row("pred/" + images_name, node_x, node_y, node_z, diam, X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio)

# Re-Write out the annotations.txt CSV file.
csvFileObj = open(os.path.join(output_path, "annotations.csv"), 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()
