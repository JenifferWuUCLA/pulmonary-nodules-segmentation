import csv, os

csv_path = "/home/ucla/Downloads/tianchi-2D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
statistics_cmp_file = os.path.join(csv_path, "statistics_cmp.csv")
statistics_sign_file = os.path.join(csv_path, "statistics_sign.csv")

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
    new_row.append(diameter_mm_error)
    new_row.append(coordX_error)
    new_row.append(coordZ_error)

    new_row.append(Y_error_ratio)
    new_row.append(diam_error_ratio)
    new_row.append(X_error_ratio)
    new_row.append(Z_error_ratio)

    new_row.append(pred_coordX)
    new_row.append(pred_coordY)
    new_row.append(pred_coordZ)
    new_row.append(pred_diameter_mm)

    csvRows.append(new_row)

########################################################################################################################

# Read the statistics_cmp.csv in (skipping first row).
stat_csvRows = []
csvFileObj = open(statistics_cmp_file)
readerObj = csv.reader(csvFileObj)
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row
    stat_csvRows.append(row)
csvFileObj.close()

csv_row("0_seriesuid", "true_coordX", "true_coordX", "true_coordZ", "true_diameter_mm",
        "avg_error", "avg_error_ratio",
        "coordX-error", "coordY-error", "coordZ-error", "diameter_mm-error",
        "X_error_ratio", "Y_error_ratio", "Z_error_ratio", "diam_error_ratio",
        "pred_coordX", "pred_coordY", "pred_coordZ", "pred_diameter_mm")

for stat_row in stat_csvRows:
    seriesuid = stat_row[0]

    true_coordX = float(stat_row[1])
    true_coordY = float(stat_row[2])
    true_coordZ = float(stat_row[3])
    true_diameter_mm = float(stat_row[4])

    pred_coordX = float(stat_row[5])
    pred_coordY = float(stat_row[6])
    pred_coordZ = float(stat_row[7])
    pred_diameter_mm = float(stat_row[8])

    coordZ_error = abs(true_coordZ - pred_coordZ)
    diameter_mm_error = abs(true_diameter_mm - pred_diameter_mm)
    Z_error_ratio = float(coordZ_error) / float(true_diameter_mm)
    diam_error_ratio = float(diameter_mm_error) / float(true_diameter_mm)

    pred_coordX_positive = abs(pred_coordX) * (1.0)
    pred_coordX_negative = abs(pred_coordX) * (-1.0)

    pred_coordY_positive = abs(pred_coordY) * (1.0)
    pred_coordY_negative = abs(pred_coordY) * (-1.0)

    coordX_error_positive = abs(true_coordX - pred_coordX_positive)
    coordX_error_negative = abs(true_coordX - pred_coordX_negative)

    coordY_error_positive = abs(true_coordY - pred_coordY_positive)
    coordY_error_negative = abs(true_coordY - pred_coordY_negative)

    X_error_ratio_positive = float(coordX_error_positive) / float(true_diameter_mm)
    X_error_ratio_negative = float(coordX_error_negative) / float(true_diameter_mm)

    Y_error_ratio_positive = float(coordY_error_positive) / float(true_diameter_mm)
    Y_error_ratio_negative = float(coordY_error_negative) / float(true_diameter_mm)

    # pred_coordX positive and pred_coordY positive
    avg_error_pp = (float)((coordX_error_positive + coordY_error_positive + coordZ_error) / 3)
    avg_error_ratio_pp = (float)((X_error_ratio_positive + Y_error_ratio_positive + Z_error_ratio) / 3)
    csv_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm,
            avg_error_pp, avg_error_ratio_pp,
            coordX_error_positive, coordY_error_positive, coordZ_error, diameter_mm_error,
            X_error_ratio_positive, Y_error_ratio_positive, Z_error_ratio, diam_error_ratio,
            pred_coordX_positive, pred_coordY_positive, pred_coordZ, pred_diameter_mm)

    # pred_coordX positive and pred_coordY negative
    avg_error_pn = (float)((coordX_error_positive + coordY_error_negative + coordZ_error) / 3)
    avg_error_ratio_pn = (float)((X_error_ratio_positive + Y_error_ratio_negative + Z_error_ratio) / 3)
    csv_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm,
            avg_error_pn, avg_error_ratio_pn,
            coordX_error_positive, coordY_error_negative, coordZ_error, diameter_mm_error,
            X_error_ratio_positive, Y_error_ratio_negative, Z_error_ratio, diam_error_ratio,
            pred_coordX_positive, pred_coordY_negative, pred_coordZ, pred_diameter_mm)

    # pred_coordX negative and pred_coordY positive
    avg_error_np = (float)((coordX_error_negative + coordY_error_positive + coordZ_error) / 3)
    avg_error_ratio_np = (float)((X_error_ratio_negative + Y_error_ratio_positive + Z_error_ratio) / 3)
    csv_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm,
            avg_error_np, avg_error_ratio_np,
            coordX_error_negative, coordY_error_positive, coordZ_error, diameter_mm_error,
            X_error_ratio_negative, Y_error_ratio_positive, Z_error_ratio, diam_error_ratio,
            pred_coordX_negative, pred_coordY_positive, pred_coordZ, pred_diameter_mm)

    # pred_coordX negative and pred_coordY negative
    avg_error_nn = (float)((coordX_error_negative + coordY_error_negative + coordZ_error) / 3)
    avg_error_ratio_nn = (float)((X_error_ratio_negative + Y_error_ratio_negative + Z_error_ratio) / 3)
    csv_row(seriesuid, true_coordX, true_coordY, true_coordZ, true_diameter_mm,
            avg_error_nn, avg_error_ratio_nn,
            coordX_error_negative, coordY_error_negative, coordZ_error, diameter_mm_error,
            X_error_ratio_negative, Y_error_ratio_negative, Z_error_ratio, diam_error_ratio,
            pred_coordX_negative, pred_coordY_negative, pred_coordZ, pred_diameter_mm)

# Write out the statistics_sign file.
print(statistics_sign_file)
csvFileObj = open(statistics_sign_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()

f = open(statistics_sign_file)
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

csvFileObj = open(statistics_sign_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in result:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()