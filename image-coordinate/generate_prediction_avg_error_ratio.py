import csv, os

csv_path = "/home/ucla/Downloads/tianchi-2D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
statistics_error_ratios_file = os.path.join(csv_path, "statistics_error_ratios.csv")
prediction_avg_error_ratios_file = os.path.join(csv_path, "prediction_avg_error_ratios.csv")

########################################################################################################################
csvRows = []


def csv_avg_error_ratios_row(X_avg_error_ratio, Y_avg_error_ratio, Z_avg_error_ratio, diam_avg_error_ratio):
    new_row = []

    new_row.append(X_avg_error_ratio)
    new_row.append(Y_avg_error_ratio)
    new_row.append(Z_avg_error_ratio)
    new_row.append(diam_avg_error_ratio)

    csvRows.append(new_row)


########################################################################################################################

# Read the statistics_error_ratios.csv in (skipping first row).
stat_csvRows = []
csvFileObj = open(statistics_error_ratios_file)
readerObj = csv.reader(csvFileObj)
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row
    stat_csvRows.append(row)
csvFileObj.close()

csv_avg_error_ratios_row("X_avg_error_ratio", "Y_avg_error_ratio", "Z_avg_error_ratio", "diam_avg_error_ratio")

sum_X_error_ratio, sum_Y_error_ratio, sum_Z_error_ratio, sum_diam_error_ratio = 0.0, 0.0, 0.0, 0.0
count = 0
for stat_row in stat_csvRows:
    seriesuid = stat_row[0]

    X_error_ratio = stat_row[5]
    Y_error_ratio = stat_row[6]
    Z_error_ratio = stat_row[7]
    diam_error_ratio = stat_row[8]

    sum_X_error_ratio += (float)(X_error_ratio)
    sum_Y_error_ratio += (float)(Y_error_ratio)
    sum_Z_error_ratio += (float)(Z_error_ratio)
    sum_diam_error_ratio += (float)(diam_error_ratio)

    count += 1

avg_X_error_ratio = (float)(sum_X_error_ratio / count)
avg_Y_error_ratio = (float)(sum_Y_error_ratio / count)
avg_Z_error_ratio = (float)(sum_Z_error_ratio / count)
avg_diam_error_ratio = (float)(sum_diam_error_ratio / count)

csv_avg_error_ratios_row(avg_X_error_ratio, avg_Y_error_ratio, avg_Z_error_ratio, avg_diam_error_ratio)


# Write out the prediction_avg_error_ratios.csv file.
print(prediction_avg_error_ratios_file)
csvFileObj = open(prediction_avg_error_ratios_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()