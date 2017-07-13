#!/usr/bin/python

import csv, os
from glob import glob


# subset = "data_set/"
tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
# tianchi_path = "/home/jenifferwu/LUNA2016/"
# tianchi_subset_path = tianchi_path + subset

# out_subset = "nerve-mine-2D/"
output_path = "/home/ucla/Downloads/tianchi-2D/"
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

nodules_coordinate_file = output_path + "image-coordinate/imgs_mask_test_coordinate.csv"

lungs_coordinate_file = output_path + "data_images/test/csv/imgs_mask_test_coordinate.csv"

annotations_file = output_path + "data_images/test/csv/imgs_mask_test_annotations.csv"

csvRows = []


def csv_row(seriesuid, coordX, coordY, coordZ, diameter_mm):
    csvRows[:] = []
    new_row = []
    new_row.append(seriesuid)
    new_row.append(coordX)
    new_row.append(coordY)
    new_row.append(coordZ)
    new_row.append(diameter_mm)
    csvRows.append(new_row)


def get_lungs_nodules(nodules_csvRows, lungs_csvRows):
    i = 0
    for nodule_row in nodules_csvRows:
        seriesuid = nodule_row['seriesuid']
        nodule_coordZ, nodule_coordX, nodule_diameter_mm = nodule_row["coordX"], nodule_row["coordY"], nodule_row["diameter_mm"]
        # nodule_coordX, nodule_coordY, nodule_diameter_mm = "-88.20063783", "63.04192832", "7.734742324"
        # print(nodule_coordX, nodule_coordZ, nodule_diameter_mm)
        if i == 0:
            min_nodule_diameter_mm = nodule_diameter_mm
        j = 0
        for lung_row in lungs_csvRows:
            lung_coordX, lung_coordY, lung_coordZ = lung_row["coordX"], lung_row["coordY"], lung_row["coordZ"]
            # print(lung_coordX, lung_coordY, lung_coordZ)
            # print(nodule_coordX, lung_coordX, abs(float(nodule_coordX) - float(lung_coordX)))
            # print(nodule_coordY, lung_coordY, abs(float(nodule_coordY) - float(lung_coordY)))
            if j == 0:
                min_lung_coordY = lung_coordY
            if abs(abs(float(nodule_coordX)) - float(lung_coordX)) <= 10 and abs(abs(float(nodule_coordZ)) - float(lung_coordZ)) <= 10:
                # print(i, j, "min_nodule_diameter_mm: %s" % min_nodule_diameter_mm, "nodule_diameter_mm: %s" % nodule_diameter_mm, (float(min_nodule_diameter_mm) >= float(nodule_diameter_mm)))
                # print(i, j, min_lung_coordY, lung_coordY, (min_lung_coordY >= lung_coordY))
                if float(min_nodule_diameter_mm) >= float(nodule_diameter_mm) and float(min_lung_coordY) >= float(lung_coordY):
                    min_nodule_diameter_mm, min_lung_coordY = nodule_diameter_mm, lung_coordY
                    # print(i, j, "min_nodule_diameter_mm: %s" % min_nodule_diameter_mm, "min_lung_coordY: %s" % min_lung_coordY)
                    csv_row(seriesuid, nodule_coordX, lung_coordY, nodule_coordZ, nodule_diameter_mm)
            j += 1
        i += 1


if __name__ == '__main__':
    # seriesuid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.367204840301639918160517361062"
    test_data_path = os.path.join(tianchi_path, 'test/')
    # print("train_data_path: %s" % train_data_path)
    test_images = glob(test_data_path + "*.mhd")

    for img_file in test_images:
        seriesuid = img_file.replace(test_data_path, "").replace(".mhd", "")
        # print(seriesuid)
        # Read the CSV file in (skipping first row).
        nodules_csvRows = []
        csvFileObj = open(nodules_coordinate_file)
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if row['seriesuid'].replace(".mhd", "") == seriesuid:
                # print(row)
                nodules_csvRows.append(row)

        lungs_csvRows = []
        csvFileObj = open(lungs_coordinate_file)
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if row['seriesuid'] == seriesuid:
                # print(row)
                lungs_csvRows.append(row)

        get_lungs_nodules(nodules_csvRows, lungs_csvRows)

        # print(csvRows)

        # Write out the imgs_mask_test_annotations CSV file.
        # print(os.path.join(output_path, annotations_file))
        csvFileObj = open(os.path.join(output_path, annotations_file), 'a')
        csvWriter = csv.writer(csvFileObj)
        for row in csvRows:
            # print row
            csvWriter.writerow(row)
        csvFileObj.close()
