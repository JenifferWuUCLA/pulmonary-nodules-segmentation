import imutils
import cv2
import os
from glob import glob
import csv
import shutil

try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


original_data_path = "/root/code/Data/"
# original_data_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/"

nodule_cnts_file = "nodule_cnts.csv"

subset = "train/"
# subset = "val/"
# subset = "pred/"
# subset = "caffe_data_set/"

data_clean_path = "/root/code/Data_clean/" + subset
# data_clean_path = "/home/jenifferwu/IMAGE_MASKS_DATA/z-nerve/Data_clean/"


###################################################################################
csvRows = []


def csv_row(image_name, cnts):
    new_row = []
    new_row.append(image_name)
    new_row.append(cnts)
    csvRows.append(new_row)


###################################################################################

# load the image, convert it to grayscale, blur it slightly,
# and threshold it
tmp_workspace = os.path.join(original_data_path, subset)
nodule_images = glob(tmp_workspace + "*.jpg")
# index = 0
csv_row("image_name", "cnts")
for img_file in nodule_images:
    print("img_file: %s" % img_file)
    o_image_name = img_file.replace(tmp_workspace, "")
    # new_name = o_image_name.replace(".npy", "") + ".jpg"

    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    print("o_image_name: %s" % o_image_name)
    # print("cnts: ")
    # print(cnts)
    if len(cnts) == 0:
        shutil.move(img_file, data_clean_path)
        # shutil.copy(img_file, data_clean_path)
        csv_row(o_image_name, cnts)

# Write out the nodule_cnts CSV file.
print(os.path.join(data_clean_path, nodule_cnts_file))
csvFileObj = open(os.path.join(data_clean_path, nodule_cnts_file), 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()