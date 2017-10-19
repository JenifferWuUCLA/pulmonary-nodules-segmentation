import csv, os

csv_path = "/home/ucla/Downloads/tianchi-3D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
annotations_true_file = os.path.join(csv_path, "annotations.csv")
annotations_pred_file = os.path.join(csv_path, "imgs_mask_test_coordinate.csv")
statistics_original_file = os.path.join(csv_path, "statistics_original.csv")

########################################################################################################################
csvRows = []


def csv_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm,
            avg_error, avg_error_ratio,
            coordX_error, coordY_error, coordZ_error, diameter_mm_error,
            X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio,
            pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm):
    new_row = []

    new_row.append(seriesuid)

    new_row.append(true_coordX)
    new_row.append(true_coordY)
    new_row.append(true_coordZ)
    new_row.append(true_diameter_mm)

    new_row.append(avg_error)
    new_row.append(avg_error_ratio)

    new_row.append(coordY_error)
    new_row.append(coordX_error)
    new_row.append(coordZ_error)
    new_row.append(diameter_mm_error)

    new_row.append(Y_error_ratio)
    new_row.append(X_error_ratio)
    new_row.append(Z_error_ratio)
    new_row.append(diam_error_ratio)

    new_row.append(pred_coordX)
    new_row.append(pred_coordY)
    new_row.append(pred_coordZ)
    new_row.append(pred_diameter_mm)

    csvRows.append(new_row)


########################################################################################################################

# Read the annotations.csv in (skipping first row).
true_csvRows = []
csvFileObj = open(annotations_true_file)
readerObj = csv.reader(csvFileObj)
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row
    true_csvRows.append(row)
csvFileObj.close()

# Read the imgs_mask_test_coordinate.csv in.
pred_csvRows = []
csvFileObj = open(annotations_pred_file)
readerObj = csv.reader(csvFileObj)
for row in readerObj:
    pred_csvRows.append(row)
csvFileObj.close()

csv_row("0_seriesuid", "true_coordX", "true_coordX", "true_coordZ", "true_diameter_mm",
        "avg_error", "avg_error_ratio",
        "coordX-error", "coordY-error", "coordZ-error", "diameter_mm-error",
        "X_error_ratio", "Y_error_ratio", "Z_error_ratio", "diam_error_ratio",
        "pred_coordX", "pred_coordY", "pred_coordZ", "pred_diameter_mm")
for true_row in true_csvRows:
    # print("true_row: ")
    # print(true_row)
    true_seriesuid = true_row[0]
    true_coordX = true_row[1]
    true_coordY = true_row[2]
    true_coordZ = true_row[3]
    true_diameter_mm = true_row[4]
    # print("True value: ")
    # print(true_seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm)
    for pred_row in pred_csvRows:
        # print("pred_row: ")
        # print(pred_row)
        pred_seriesuid = pred_row[0]
        pred_coordX = pred_row[1]
        pred_coordY = pred_row[2]
        pred_coordZ = pred_row[3]
        pred_diameter_mm = pred_row[4]
        # print("Prediction value: ")
        # print(pred_seriesuid, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm)
        if true_seriesuid == pred_seriesuid:
            coordX_error = abs(abs(float(true_coordX)) - abs(float(pred_coordX)))
            coordY_error = abs(abs(float(true_coordY)) - abs(float(pred_coordY)))
            coordZ_error = abs(abs(float(true_coordZ)) - abs(float(pred_coordZ)))
            if float(coordZ_error) > 5:
                continue
            diameter_mm_error = abs(abs(float(true_diameter_mm)) - abs(float(pred_diameter_mm)))

            X_error_ratio = float(coordX_error) / float(true_diameter_mm)
            Y_error_ratio = float(coordY_error) / float(true_diameter_mm)
            Z_error_ratio = float(coordZ_error) / float(true_diameter_mm)
            diam_error_ratio = float(diameter_mm_error) / float(true_diameter_mm)

            avg_error = (float)((coordX_error + coordY_error + coordZ_error) / 3)
            avg_error_ratio = (float)((X_error_ratio + Y_error_ratio + Z_error_ratio) / 3)

            csv_row(true_seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm,
                    avg_error, avg_error_ratio,
                    coordX_error, coordY_error, coordZ_error, diameter_mm_error,
                    X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio,
                    pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm)

# Write out the statistics file.
print(statistics_original_file)
csvFileObj = open(statistics_original_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()

f = open(statistics_original_file)
result = []
iter_f = iter(f)  # Iterate through each line in a file with an iterator
index = 0
for line in iter_f:
    row = line.split(",")
    new_row = []

    new_row.append(row[0])

    new_row.append(row[1])
    new_row.append(row[2])
    new_row.append(row[3])
    new_row.append(row[4])

    index += 1
    if index == 1:
        new_row.append(row[5])
        new_row.append(row[6])
        new_row.append(row[7])
        new_row.append(row[8])
        new_row.append(row[9])
        new_row.append(row[10])
        new_row.append(row[11])
        new_row.append(row[12])
        new_row.append(row[13])
        new_row.append(row[14])
    else:
        new_row.append(float(row[5]))
        new_row.append(float(row[6]))
        new_row.append(float(row[7]))
        new_row.append(float(row[8]))
        new_row.append(float(row[9]))
        new_row.append(float(row[10]))
        new_row.append(float(row[11]))
        new_row.append(float(row[12]))
        new_row.append(float(row[13]))
        new_row.append(float(row[14]))

    new_row.append(row[15])
    new_row.append(row[16])
    new_row.append(row[17])
    new_row.append((row[18]).replace("\r\n", ""))

    result.append(new_row)
f.close()

result.sort()

csvFileObj = open(statistics_original_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in result:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()