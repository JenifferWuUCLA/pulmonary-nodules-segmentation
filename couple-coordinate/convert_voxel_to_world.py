import numpy as np
import os
import csv
import SimpleITK as sitk

try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


subset = "val_subset_all/"
# subset = "test/"
tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
# tianchi_path = "/home/jenifferwu/LUNA2016/data_set/"

test_data_path = os.path.join(tianchi_path, subset)

out_subset = "server-test-mine/"
output_path = "/home/ucla/Downloads/tianchi-2D/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

lungs_coordinate_file = output_path + "data_images/test/csv/imgs_mask_test_coordinate.csv"
lungs_coordinate_file_3D = output_path + "data_images/test/csv/imgs_mask_test_coordinate_3D.csv"


#######################################################################################################################
def voxel_2_world(voxelCoord, origin, spacing):
    stretchedVoxelCoord = voxelCoord * spacing
    worldCoord = stretchedVoxelCoord + origin
    return worldCoord


csvRows = []


def csv_row(seriesuid, coordX, coordY, coordZ, diameter_mm):
    new_row = []
    new_row.append(seriesuid)
    new_row.append(coordX)
    new_row.append(coordY)
    new_row.append(coordZ)
    new_row.append(diameter_mm)
    csvRows.append(new_row)


#######################################################################################################################

if __name__ == '__main__':
    csv_row("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm")
    # Read the CSV file in (skipping first row).
    csvRows = []
    csvFileObj = open(lungs_coordinate_file)
    readerObj = csv.reader(csvFileObj)
    for row in readerObj:
        if readerObj.line_num == 1:
            continue  # skip first row
        csvRows.append(row)
    csvFileObj.close()

    for csvRow in csvRows:
        seriesuid = csvRow["seriesuid"]
        image_name = seriesuid + ".mhd"
        original_file_name = os.path.join(test_data_path, image_name)
        itk_img = sitk.ReadImage(original_file_name)
        # load the data once
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)

        coordX, coordY, coordZ = csvRow["coordX"], csvRow["coordY"], csvRow["coordZ"]
        v_center = np.array([float(coordX), float(coordY)], float(coordZ))
        print("v_center: ")
        print(v_center)

        w_center = voxel_2_world(v_center, origin, spacing)
        print("w_center: ")
        print(w_center)

        diameter_mm = csvRow["diameter_mm"]
        csv_row(seriesuid, coordX, coordY, coordZ, "diameter_mm")

    # Write out the lungs_coordinate_file_3D CSV file.
    print(os.path.join(output_path, lungs_coordinate_file_3D))
    csvFileObj = open(os.path.join(output_path, lungs_coordinate_file_3D), 'w')
    csvWriter = csv.writer(csvFileObj)
    for row in csvRows:
        # print row
        csvWriter.writerow(row)
    csvFileObj.close()
