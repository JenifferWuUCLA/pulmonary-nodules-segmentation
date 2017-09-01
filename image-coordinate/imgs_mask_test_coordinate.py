import imutils
import cv2
import os
from glob import glob
import csv
import numpy as np
import SimpleITK as sitk

try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


# out_subset = "z-nerve/"
output_path = "/home/ucla/Downloads/tianchi-2D/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

coordinate_file = "image-coordinate-2D/imgs_mask_test_coordinate.csv"

subset = "val_subset_all/"
# subset = "test_subset_all/"
# subset = "train_dataset/"
# subset = "data_set/"
tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
# tianchi_path = "/home/jenifferwu/LUNA2016/"
# tianchi_subset_path = tianchi_path + subset

test_data_path = os.path.join(tianchi_path, subset)


###################################################################################

def voxelToWorld(voxelCoord, origin, spacing):
    voxelCoord = voxelCoord.astype(np.int32)
    w_x = spacing[0] * voxelCoord[0] + origin[0]
    w_y = spacing[1] * voxelCoord[1] + origin[1]
    w_z = spacing[2] * voxelCoord[2] + origin[2]
    worldCoord = np.array([float(w_x), float(w_y), float(w_z)])

    return worldCoord


'''
def voxel_2_world(voxelCoord, origin, spacing):
    stretchedVoxelCoord = voxelCoord * spacing
    worldCoord = stretchedVoxelCoord + origin
    return worldCoord
'''


def image_file_name(image_name):
    # Read the annotations CSV file in (skipping first row).
    if os.path.exists(os.path.join(output_path, "seriesuid_pred_image.csv")):
        csvFileObj = open(os.path.join(output_path, "seriesuid_pred_image.csv"), 'r')
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if readerObj.line_num == 1:
                continue  # skip first row
            seriesuid = row['seriesuid']
            pred_image = row['pred_image']
            if image_name == pred_image:
                csvFileObj.close()
                return seriesuid
        csvFileObj.close()


csvRows = []


def csv_row(image_name, x, y, z, radius):
    new_row = []
    new_row.append(image_name)
    new_row.append(x)
    new_row.append(y)
    new_row.append(z)
    new_row.append(radius)
    csvRows.append(new_row)


###################################################################################

# load the image, convert it to grayscale, blur it slightly,
# and threshold it
tmp_workspace = os.path.join(output_path, "image-coordinate-2D/bright_02/")
test_images = glob(tmp_workspace + "*.jpg")
# index = 0
csv_row("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm")
for img_file in test_images:
    print("img_file: %s" % img_file)
    o_image_name = img_file.replace(tmp_workspace, "")
    # new_name = o_image_name.replace(".npy", "") + ".jpg"

    # o_image_name = o_image_name.replace(".jpg", ".npy")
    print("image_name before: %s" % o_image_name)
    tmp_image_name = image_file_name(o_image_name)
    print("image_name after: %s" % tmp_image_name)
    image_name = tmp_image_name.split("_")[0] + ".mhd"
    v_z = tmp_image_name.split("_")[1]
    index = tmp_image_name.split("_")[2]

    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # print("cnts: ")
    # print(cnts)
    # loop over the contours
    for c in cnts:
        (x, y), radius = cv2.minEnclosingCircle(c)
        # center = (float(x), float(y))
        # print("center: ")
        # print(center, radius)
        radius = float(radius)
        # img = cv2.circle(image, center, radius, (0, 255, 0), 2)

        # print("image_name: %s" % image_name)
        original_file_name = os.path.join(test_data_path, image_name)
        itk_img = sitk.ReadImage(original_file_name)
        # load the data once
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
        print("file_name: ")
        print(image_name, origin, spacing)

        for i, i_z in enumerate(np.arange(int(v_z) - 1, int(v_z) + 2).clip(0, num_z - 1)):
            if i != int(index):
                continue

            v_center = np.array([float(x), float(y), float(i_z)])
            print("v_center: ")
            print(v_center, radius)

            w_center = voxelToWorld(v_center, origin, spacing)
            print("w_center: ")
            print(w_center, radius)

            csv_row(image_name.replace(".mhd", ""), w_center[0], w_center[1], w_center[2], radius)

        # index += 1

# Write out the imgs_mask_test_coordinate CSV file.
print(os.path.join(output_path, coordinate_file))
csvFileObj = open(os.path.join(output_path, coordinate_file), 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()
