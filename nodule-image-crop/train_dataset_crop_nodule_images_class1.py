import numpy as np
from glob import glob
import os
import cv2


original_output_path = "/root/code/Pulmonary_nodules_data/"
# original_output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/Pulmonary_nodules_data/"
crop_output_path = "/root/code/Pulmonary_nodules_data_crop/"
# crop_output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/Pulmonary_nodules_data_crop/"

original_nodule_path = os.path.join(original_output_path, "train/n01440011/")
crop_nodule_path = os.path.join(crop_output_path, "train/n01440011/")

train_images = glob(os.path.join(original_output_path, "train/n01440011/n01440011_*.jpg"))
for img_file in train_images:
    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.cv.BoxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # cv2.imshow("Image", image)

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = image[y1:y1+hight, x1:x1+width]

    filename = img_file.replace(original_nodule_path, "")
    cv2.imwrite(os.path.join(crop_nodule_path, filename), cropImg)
    cv2.waitKey(0)