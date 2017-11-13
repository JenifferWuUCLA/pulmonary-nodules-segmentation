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
nodule_diameter_mm_file = os.path.join(csv_path, "test_nodule_diameter_mm.csv")

############
test_data_path = os.path.join(tianchi_path, subset)
test_images = glob(test_data_path + "*.mhd")


########################################################################################################################
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


csvRows = []


def csv_nodule_diameter_mm_row(seriesuid, diameter):
    new_row = []
    new_row.append(seriesuid)
    new_row.append(diameter)
    csvRows.append(new_row)


########################################################################################################################

# The locations of the nodes
df_node = pd.read_csv(tianchi_path + "csv/test/annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(test_images, file_name))
df_node = df_node.dropna()

#####
#
# Looping over the test image files
#
for fcount, img_file in enumerate(tqdm(test_images)):
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():
            seriesuid = cur_row["seriesuid"]
            diam = cur_row["diameter_mm"]
            csv_nodule_diameter_mm_row(seriesuid, diam)

# Write out the test_nodule_diameter_mm.csv file.
print(nodule_diameter_mm_file)
csvFileObj = open(nodule_diameter_mm_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()

f = open(nodule_diameter_mm_file)
result = []
iter_f = iter(f)  # Iterate through each line in a file with an iterator
index = 0
for line in iter_f:
    row = line.split(",")
    new_row = []
    new_row.append(float(row[1].replace("\r\n", "")))
    new_row.append(row[0])
    result.append(new_row)
f.close()

result.sort()

csvFileObj = open(nodule_diameter_mm_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in result:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()