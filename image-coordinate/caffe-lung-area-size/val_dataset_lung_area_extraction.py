from __future__ import print_function, division
import csv, os
import SimpleITK as sitk
from glob import glob
import pandas as pd

try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


subset = "tianchi_val_dataset/"
# subset = "data_set/"
tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
# tianchi_path = "/home/jenifferwu/LUNA2016/"
# tianchi_subset_path = tianchi_path + subset

# out_subset = "z-nerve"
output_path = "/home/ucla/Downloads/tianchi-caffe/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

csv_path = output_path + "csv/"
lung_slice_area_size_file = os.path.join(csv_path, "val_lung_slice_area_size.csv")

############
val_data_path = os.path.join(tianchi_path, subset)
# print("val_data_path: %s" % val_data_path)
val_images = glob(val_data_path + "*.mhd")
# print(val_images)

tmp_workspace = os.path.join(output_path, "val/")
tmp_jpg_workspace = os.path.join(output_path, "ROI/val/")


########################################################################################################################
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


csvRows = []


def csv_lung_slice_area_size_row(seriesuid, height, width):
    new_row = []
    new_row.append(seriesuid)
    new_row.append(height)
    new_row.append(width)
    csvRows.append(new_row)


########################################################################################################################

# The locations of the nodes
df_node = pd.read_csv(tianchi_path + "csv/val/annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(val_images, file_name))
df_node = df_node.dropna()

#####
#
# Looping over the val image files
#
for fcount, img_file in enumerate(tqdm(val_images)):
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():
            seriesuid = cur_row["seriesuid"]
            csv_lung_slice_area_size_row(seriesuid, height, width)

# Write out the lung_slice_area_size.csv file.
print(lung_slice_area_size_file)
csvFileObj = open(lung_slice_area_size_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()
