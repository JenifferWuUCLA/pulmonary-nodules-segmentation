import csv, os

csv_path = "/home/ucla/Downloads/tianchi-3D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
suspected_sign_file = os.path.join(csv_path, "suspected_original.csv")
suspected_annotations_file = os.path.join(csv_path, "suspected_annotations.csv")
suspected_error_ratios_file = os.path.join(csv_path, "suspected_error_ratios.csv")

########################################################################################################################
csvAnnotationsRows = []


def csv_annotations_row(seriesuid, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm):
    new_row = []

    new_row.append(seriesuid)

    new_row.append(pred_coordX)
    new_row.append(pred_coordY)
    new_row.append(pred_coordZ)
    new_row.append(pred_diameter_mm)

    csvAnnotationsRows.append(new_row)


csvRows = []


def csv_error_ratios_row(seriesuid, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm,
                         X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio):
    new_row = []

    new_row.append(seriesuid)

    new_row.append(pred_coordX)
    new_row.append(pred_coordY)
    new_row.append(pred_coordZ)
    new_row.append(pred_diameter_mm)

    new_row.append(X_error_ratio)
    new_row.append(Y_error_ratio)
    new_row.append(Z_error_ratio)
    new_row.append(diam_error_ratio)

    csvRows.append(new_row)


########################################################################################################################

# Read the suspected_original.csv in (skipping first row).
stat_csvRows = []
csvFileObj = open(suspected_sign_file)
readerObj = csv.reader(csvFileObj)
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row
    stat_csvRows.append(row)
csvFileObj.close()

csv_annotations_row("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm")

csv_error_ratios_row("seriesuid", "pred_coordX", "pred_coordY", "pred_coordZ", "pred_diameter_mm", "X_error_ratio",
                     "Y_error_ratio",
                     "Z_error_ratio", "diam_error_ratio")

for stat_row in stat_csvRows:
    seriesuid = stat_row[0].split("_")[0]

    true_coordX = stat_row[1]
    true_coordY = stat_row[2]
    true_coordZ = stat_row[3]
    true_diameter_mm = stat_row[4]

    avg_error = stat_row[5]
    avg_error_ratio = stat_row[6]

    coordX_error = stat_row[7]
    coordY_error = stat_row[8]
    coordZ_error = stat_row[9]
    diameter_mm_error = stat_row[10]

    X_error_ratio = stat_row[11]
    Y_error_ratio = stat_row[12]
    Z_error_ratio = stat_row[13]
    diam_error_ratio = stat_row[14]

    pred_coordX = stat_row[15]
    pred_coordY = stat_row[16]
    pred_coordZ = stat_row[17]
    pred_diameter_mm = stat_row[18]

    csv_annotations_row(seriesuid, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm)
    csv_error_ratios_row(seriesuid, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm, X_error_ratio,
                         Y_error_ratio, Z_error_ratio, diam_error_ratio)

# Write out the suspected_annotations.csv file.
print(suspected_annotations_file)
csvFileObj = open(suspected_annotations_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvAnnotationsRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()

# Write out the suspected_error_ratios.csv file.
print(suspected_error_ratios_file)
csvFileObj = open(suspected_error_ratios_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()