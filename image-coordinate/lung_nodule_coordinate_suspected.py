import csv, os

csv_path = "/home/ucla/Downloads/tianchi-2D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
statistics_original_file = os.path.join(csv_path, "statistics_original.csv")
statistics_cmp_file = os.path.join(csv_path, "statistics_cmp.csv")
statistics_file = os.path.join(csv_path, "statistics.csv")
statistics_MSE_file = os.path.join(csv_path, "statistics_MSE.csv")

########################################################################################################################
csvCmpRows = []


def csv_cmp_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm):
    new_row = []

    new_row.append(seriesuid)

    new_row.append(true_coordX)
    new_row.append(true_coordY)
    new_row.append(true_coordZ)
    new_row.append(true_diameter_mm)

    new_row.append(pred_coordX)
    new_row.append(pred_coordY)
    new_row.append(pred_coordZ)
    new_row.append(pred_diameter_mm)

    csvCmpRows.append(new_row)


csvRows = []


def csv_row(seriesuid, avg_error, avg_error_ratio, coordX_error, coordY_error, coordZ_error, diameter_mm_error, X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio):
    new_row = []

    new_row.append(seriesuid)

    new_row.append(avg_error)
    new_row.append(avg_error_ratio)

    new_row.append(coordX_error)
    new_row.append(coordY_error)
    new_row.append(coordZ_error)
    new_row.append(diameter_mm_error)

    new_row.append(X_error_ratio)
    new_row.append(Y_error_ratio)
    new_row.append(Z_error_ratio)
    new_row.append(diam_error_ratio)

    csvRows.append(new_row)


csvMSERows = []


def csv_MSE_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm):
    new_row = []

    new_row.append(seriesuid)

    new_row.append(abs(float(true_coordX)))
    new_row.append(abs(float(true_coordY)))
    new_row.append(abs(float(true_coordZ)))
    new_row.append(abs(float(true_diameter_mm)))

    new_row.append(abs(float(pred_coordX)))
    new_row.append(abs(float(pred_coordY)))
    new_row.append(abs(float(pred_coordZ)))
    new_row.append(abs(float(pred_diameter_mm)))

    csvMSERows.append(new_row)

########################################################################################################################

# Read the statistics_original.csv in (skipping first row).
stat_csvRows = []
csvFileObj = open(statistics_original_file)
readerObj = csv.reader(csvFileObj)
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row
    stat_csvRows.append(row)
csvFileObj.close()

csv_cmp_row("seriesuid", "true_coordX", "true_coordY", "true_coordZ", "true_diameter_mm", "pred_coordX", "pred_coordY", "pred_coordZ", "pred_diameter_mm")

csv_row("seriesuid", "avg_error", "avg_error_ratio", "coordX-error", "coordY-error", "coordZ-error", "diameter_mm-error", "X_error_ratio", "Y_error_ratio", "Z_error_ratio", "diam_error_ratio")

# csv_MSE_row("seriesuid", "true_coordX", "true_coordY", "true_coordZ", "true_diameter_mm", "pred_coordX", "pred_coordY", "pred_coordZ", "pred_diameter_mm")

last_seriesuid, last_true_coordX, last_true_coordY, last_true_coordZ, last_true_diameter_mm = "", "", "", "", ""
for stat_row in stat_csvRows:
    seriesuid = stat_row[0]

    true_coordX = stat_row[1]
    true_coordY = stat_row[2]
    true_coordZ = stat_row[3]
    true_diameter_mm = stat_row[4]

    avg_error = stat_row[5]
    avg_error_ratio = stat_row[6]

    coordX_error = stat_row[9]
    coordY_error = stat_row[7]
    coordZ_error = stat_row[10]
    diameter_mm_error = stat_row[8]

    X_error_ratio = stat_row[13]
    Y_error_ratio = stat_row[11]
    Z_error_ratio = stat_row[14]
    diam_error_ratio = stat_row[12]

    pred_coordX = stat_row[15]
    pred_coordY = stat_row[16]
    pred_coordZ = stat_row[17]
    pred_diameter_mm = stat_row[18]

    condition_1 = (seriesuid != last_seriesuid)
    condition_2 = (true_coordX != last_true_coordX) or (true_coordY != last_true_coordY) or (true_coordZ != last_true_coordZ) or (true_diameter_mm != last_true_diameter_mm)

    if condition_1 or condition_2:
        csv_cmp_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm)
        csv_row(seriesuid, avg_error, avg_error_ratio, coordX_error, coordY_error, coordZ_error, diameter_mm_error, X_error_ratio, Y_error_ratio, Z_error_ratio, diam_error_ratio)
        csv_MSE_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm)

    last_seriesuid, last_true_coordX, last_true_coordY, last_true_coordZ, last_true_diameter_mm = seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm


# Write out the statistics_cmp.csv file.
print(statistics_cmp_file)
csvCmpFileObj = open(statistics_cmp_file, 'w')
csvCmpWriter = csv.writer(csvCmpFileObj)
for row in csvCmpRows:
    # print row
    csvCmpWriter.writerow(row)
csvFileObj.close()

# Write out the statistics.csv file.
print(statistics_file)
csvFileObj = open(statistics_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()

# Write out the statistics_MSE.csv file.
print(statistics_MSE_file)
csvMSEFileObj = open(statistics_MSE_file, 'w')
csvMSEWriter = csv.writer(csvMSEFileObj)
for row in csvMSERows:
    # print row
    csvMSEWriter.writerow(row)
csvFileObj.close()