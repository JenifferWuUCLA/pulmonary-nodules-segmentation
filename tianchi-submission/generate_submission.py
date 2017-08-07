import os
import csv

subset = "val_subset_all/"
# subset = "server-test-2D/"
output_path = "/home/ucla/Downloads/tianchi/" + subset
# output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + subset

submission_file = "instant-recognition/submission.csv"

test_annotations_file = "csv/test/annotations.csv"

pulmonary_nodule_probability_file = "pulmonary_nodule_probability.csv"

dict_annotations, dict_probabilities = {}, {}

csvRows = []


def csv_row(seriesuid, coordX, coordY, coordZ, diameter_mm, probability):
    new_row = []
    new_row.append(seriesuid)
    new_row.append(coordX)
    new_row.append(coordY)
    new_row.append(coordZ)
    new_row.append(diameter_mm)
    new_row.append(probability)
    csvRows.append(new_row)


csv_row("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "probability")


def get_dict_annotations():
    # Read the annotations CSV file in (skipping first row).
    if os.path.exists(os.path.join(os.getcwd(), test_annotations_file)):
        csvFileObj = open(os.path.join(os.getcwd(), test_annotations_file), 'r')
        # print(os.path.join(os.getcwd(), test_annotations_file))
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if readerObj.line_num == 1:
                continue  # skip first row
            seriesuid = row['seriesuid']
            coordX, coordY, coordZ, diameter_mm = row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']
            # print("seriesuid, coordX, coordY, coordZ, diameter_mm: %s, %s, %s, %s, %s" % (seriesuid, coordX, coordY, coordZ, diameter_mm))
            dict_annotations[seriesuid] = (coordX, coordY, coordZ, diameter_mm)
        csvFileObj.close()
        return dict_annotations


def get_dict_probability():
    # Read the pulmonary_nodule_probability CSV file in (skipping first row).
    # print(os.path.join(os.getcwd(), pulmonary_nodule_probability_file))
    if os.path.exists(os.path.join(os.getcwd(), pulmonary_nodule_probability_file)):
        csvFileObj = open(os.path.join(os.getcwd(), pulmonary_nodule_probability_file), 'r')
        # print(os.path.join(os.getcwd(), pulmonary_nodule_probability_file))
        readerObj = csv.DictReader(csvFileObj)
        for row in readerObj:
            if readerObj.line_num == 1:
                continue  # skip first row
            seriesuid = row['seriesuid']
            probability, label = row['probability'], row['label']
            # print("seriesuid, probability, label: %s, %s, %s" % (seriesuid, probability, label))
            synset = label.split(" ")[0]
            # print("synset: %s" % synset)
            if synset == "n01440011":
                dict_probabilities[seriesuid] = (probability, label)
        csvFileObj.close()
        return dict_probabilities


if __name__ == '__main__':
    dict_annotations = get_dict_annotations()
    dict_probabilities = get_dict_probability()
    # print("dict_annotations: %s" % str(len(dict_annotations)))
    # print("dict_probabilities: %s" % str(len(dict_probabilities)))

    for annotations in dict_annotations.items():
        annotations_seriesuid = annotations[0]
        a_seriesuid = annotations_seriesuid
        annotations_tuple = annotations[1]
        # print("annotations_seriesuid, annotations_tuple: ")
        # print(annotations_seriesuid, annotations_tuple)
        coordX, coordY, coordZ, diameter_mm = annotations_tuple[0], annotations_tuple[1], annotations_tuple[2], annotations_tuple[3]
        for probabilities in dict_probabilities.items():
            probabilities_seriesuid = probabilities[0]
            p_seriesuid = probabilities_seriesuid.split("_")[0]
            if a_seriesuid == p_seriesuid:
                probabilities_tuple = probabilities[1]
                # print("probabilities_seriesuid, probabilities_tuple: ")
                # print(probabilities_seriesuid, probabilities_tuple)
                probability, label = probabilities_tuple[0], probabilities_tuple[1]
                synset = label.split(" ")[0]
                if synset == "n01440011":
                    csv_row(p_seriesuid, coordX, coordY, coordZ, diameter_mm, probability)
                    break

    # Write out the pulmonary_nodule_probability CSV file.
    print(os.path.join(output_path, submission_file))
    csvFileObj = open(os.path.join(output_path, submission_file), 'w')
    csvWriter = csv.writer(csvFileObj)
    for row in csvRows:
        # print row
        csvWriter.writerow(row)
    csvFileObj.close()