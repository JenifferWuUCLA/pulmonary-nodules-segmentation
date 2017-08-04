import os
import csv

subset = "val_subset_all/"
output_path = "/home/ucla/Downloads/tianchi/" + subset

submission_file = "instant-recognition/submission.csv"

test_annotations_file = "csv/test/annotations.csv"

pulmonary_nodule_probability_file = "pulmonary_nodule_probability"

dict_annotations, dict_probabilities = {}, {}

csvRows = []


def csv_row(image_name, probability, label):
    new_row = []
    new_row.append(image_name)
    new_row.append(probability)
    new_row.append(label)
    csvRows.append(new_row)


csv_row("seriesuid", "coordX", "coordY", "diameter_mm", "probability")


def get_dict_annotations():
    # Read the annotations CSV file in (skipping first row).
    if os.path.exists(os.path.join(".", test_annotations_file)):
        csvFileObj = open(os.path.join(".", test_annotations_file), 'r')
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if readerObj.line_num == 1:
                continue  # skip first row
            seriesuid = row['seriesuid']
            coordX, coordY, coordZ, diameter_mm = row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']
            dict_annotations[seriesuid] = (coordX, coordY, coordZ)
        csvFileObj.close()


def get_dict_probability():
    # Read the pulmonary_nodule_probability CSV file in (skipping first row).
    if os.path.exists(os.path.join(".", pulmonary_nodule_probability_file)):
        csvFileObj = open(os.path.join(".", pulmonary_nodule_probability_file), 'r')
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if readerObj.line_num == 1:
                continue  # skip first row
            seriesuid = row['seriesuid']
            probability, label = row['probability'], row['label']
            dict_probabilities[seriesuid] = (probability, label)
        csvFileObj.close()


if __name__ == '__main__':
    for annotations in dict_annotations.items():
        annotations_seriesuid = annotations[0]
        annotations_tuple = annotations[1]
        coordX, coordY, coordZ, diameter_mm = annotations_tuple[0], annotations_tuple[1], annotations_tuple[2], annotations_tuple[3]
        for probabilities in dict_probabilities.items():
            probabilities_seriesuid = probabilities[0]
            probabilities_tuple = probabilities[1]
            probability, label = probabilities_tuple[0], probabilities_tuple[1]
            synset = label.split(" ")[0]
            if synset == "n01440011":
                csv_row(probabilities_seriesuid, coordX, coordY, coordZ, diameter_mm, probability)
                break

    # Write out the pulmonary_nodule_probability CSV file.
    print(os.path.join(output_path, submission_file))
    csvFileObj = open(os.path.join(output_path, submission_file), 'w')
    csvWriter = csv.writer(csvFileObj)
    for row in csvRows:
        # print row
        csvWriter.writerow(row)
    csvFileObj.close()