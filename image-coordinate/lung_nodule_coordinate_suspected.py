import csv, os

csv_path = "/home/ucla/Downloads/tianchi-2D/csv"
# csv_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/csv"
statistics_original_file = os.path.join(csv_path, "statistics_original.csv")
statistics_file = os.path.join(csv_path, "statistics.csv")

########################################################################################################################
csvRows = []


def csv_row(seriesuid, coordX_error, coordY_error, coordZ_error, diameter_mm_error):
    new_row = []
    new_row.append(seriesuid)
    new_row.append(coordX_error)
    new_row.append(coordY_error)
    new_row.append(coordZ_error)
    new_row.append(diameter_mm_error)
    csvRows.append(new_row)


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

csv_row("seriesuid", "coordX-error", "coordY-error", "coordZ-error", "diameter_mm-error")

last_seriesuid, last_pred_coordX, last_pred_coordY, last_pred_coordZ, last_pred_diameter_mm = "", "", "", "", ""
for stat_row in stat_csvRows:
    seriesuid = stat_row[0]
    pred_coordX = stat_row[1]
    pred_coordY = stat_row[2]
    pred_coordZ = stat_row[3]
    pred_diameter_mm = stat_row[4]
    avg_error = stat_row[5]
    coordX_error = stat_row[6]
    coordY_error = stat_row[7]
    coordZ_error = stat_row[8]
    diameter_mm_error = stat_row[9]

    if seriesuid != last_seriesuid or pred_coordX != last_pred_coordX or pred_coordY != last_pred_coordY or pred_coordZ != last_pred_coordZ or pred_diameter_mm != last_pred_diameter_mm:
        csv_row(seriesuid, coordX_error, coordY_error, coordZ_error, diameter_mm_error)

    last_seriesuid, last_pred_coordX, last_pred_coordY, last_pred_coordZ, last_pred_diameter_mm = seriesuid, pred_coordX, pred_coordY, pred_coordZ, pred_diameter_mm

# Write out the statistics file.
print(statistics_file)
csvFileObj = open(statistics_file, 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()